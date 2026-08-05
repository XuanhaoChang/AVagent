"""Orchestration for the Auralis audio-visual forensic agent."""

from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agents.auralis.constrained_asr import (
    constrained_asr_issues,
    filter_contradicted_judge_issues,
)
from agents.auralis.schemas import (
    AuralisEvidence,
    AuralisInput,
    AuralisResult,
)
from tools.media.ffmpeg import extract_audio_wav, probe_video
from tools.speech_subtitle_alignment.tool import check_speech_subtitle_alignment
from tools.speech_subtitle_alignment.schemas import AlignmentResult
from tools.speech_transcription.schemas import SpeechTranscript
from tools.subtitle_extraction.schemas import SubtitleSegment, SubtitleTrack
from tools.subtitle_extraction.tool import (
    extract_subtitles as run_subtitle_extraction,
    subtitle_evidence_for_judge,
)


Judge = Callable[[AuralisInput, AuralisEvidence], Sequence[Mapping[str, Any]]]
PromptCandidateScorer = Callable[
    [Path, str, SpeechTranscript],
    Mapping[str, Any],
]
PromptAwareTranscriber = Callable[[Path, str], SpeechTranscript]


_ISSUE_TIME_RANGE = re.compile(
    r"^\s*(?P<start>\d+(?:\.\d+)?)s?\s*-\s*"
    r"(?P<end>\d+(?:\.\d+)?)s?\s*$"
)


def _no_judge(_agent_input: AuralisInput, _evidence: AuralisEvidence):
    return ()


def deterministic_alignment_issues(
    alignment: AlignmentResult,
) -> tuple[Mapping[str, Any], ...]:
    """Convert only high-precision deterministic ASR/OCR diffs into issues."""

    issues: list[Mapping[str, Any]] = []
    for item in alignment.issues:
        if (
            item.method
            not in {"localized_asr_ocr", "numeric_timeline_alignment"}
            or item.confidence != "high"
        ):
            continue
        issues.append(
            {
                "可定位性": "否",
                "置信度": "高",
                "问题说明": (
                    "预期烧录字幕应与实际语音一致；"
                    f"ASR 实际语音为“{item.speech_text}”，"
                    f"OCR 实际字幕为“{item.subtitle_text}”，"
                    f"{item.difference}；发生在 "
                    f"{item.start_sec:.2f}s - {item.end_sec:.2f}s。"
                ),
                "问题类型": "文字质量问题",
                "时间区间": f"{item.start_sec:.2f}s - {item.end_sec:.2f}s",
                "关键帧秒": "",
                "BBox": "",
            }
        )
    return tuple(issues)


def _time_range(issue: Mapping[str, Any]) -> tuple[float, float] | None:
    match = _ISSUE_TIME_RANGE.fullmatch(str(issue.get("时间区间") or ""))
    if match is None:
        return None
    start = float(match.group("start"))
    end = float(match.group("end"))
    if end <= start:
        return None
    return start, end


def _overlaps(start: float, end: float, other_start: float, other_end: float) -> bool:
    return min(end, other_end) > max(start, other_start)


def filter_unverified_ocr_judge_issues(
    trusted_subtitles: SubtitleTrack,
    alignment: AlignmentResult,
    rejected_singletons: Sequence[SubtitleSegment],
    issues: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    """Veto text defects whose only local support is singleton-frame OCR."""

    if not rejected_singletons:
        return tuple(issues), ()
    kept: list[Mapping[str, Any]] = []
    vetoed: list[Mapping[str, Any]] = []
    for issue in issues:
        interval = _time_range(issue)
        if issue.get("问题类型") != "文字质量问题" or interval is None:
            kept.append(issue)
            continue
        start, end = interval
        rejected = [
            segment
            for segment in rejected_singletons
            if _overlaps(start, end, segment.start_sec, segment.end_sec)
        ]
        if not rejected:
            kept.append(issue)
            continue
        has_trusted_ocr = any(
            _overlaps(start, end, segment.start_sec, segment.end_sec)
            for segment in trusted_subtitles.segments
        )
        has_alignment_support = any(
            _overlaps(start, end, item.start_sec, item.end_sec)
            for item in alignment.issues
        )
        if has_trusted_ocr or has_alignment_support:
            kept.append(issue)
            continue
        vetoed.append(
            {
                "issue": dict(issue),
                "reason": "only_unverified_single_frame_single_character_ocr",
                "ocr_candidates": [
                    {
                        "start_sec": segment.start_sec,
                        "end_sec": segment.end_sec,
                        "text": segment.text,
                        "bbox": list(segment.bbox),
                        "confidence": segment.confidence,
                        "source": segment.source,
                    }
                    for segment in rejected
                ],
            }
        )
    return tuple(kept), tuple(vetoed)


def filter_acoustically_contradicted_binding_issues(
    transcript: SpeechTranscript,
    issues: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    """Veto same-voice claims contradicted by anchored fine-grained turns."""

    metadata = transcript.metadata
    binding = metadata.get("speaker_binding_evidence", {})
    if not isinstance(binding, Mapping) or binding.get("status") != "fine_grained_turns":
        return tuple(issues), ()
    alignments = [
        item
        for item in binding.get("prompt_turn_alignment", ())
        if isinstance(item, Mapping)
        and item.get("status") == "anchored"
        and len(item.get("actual_speakers", ())) == 1
    ]
    same_voice_markers = (
        "同一声纹",
        "同一个声纹",
        "声线完全一致",
        "声纹标签未发生变化",
        "speaker标签未发生变化",
        "spk标签未发生变化",
        "由同一人发出",
    )
    binding_markers = ("绑定", "归属", "声纹", "说话人", "配音主体")
    kept: list[Mapping[str, Any]] = []
    vetoed: list[Mapping[str, Any]] = []
    for issue in issues:
        description = str(issue.get("问题说明") or "")
        if not (
            issue.get("问题类型") == "音频质量问题"
            and any(marker in description for marker in binding_markers)
            and any(marker in description for marker in same_voice_markers)
        ):
            kept.append(issue)
            continue
        mentioned = [
            item
            for item in alignments
            if str(item.get("role") or "")
            and str(item.get("role")) in description
        ]
        roles = {str(item.get("role")) for item in mentioned}
        speakers = {
            str(item["actual_speakers"][0])
            for item in mentioned
            if item.get("actual_speakers")
        }
        if len(roles) >= 2 and len(speakers) >= 2:
            vetoed.append(
                {
                    "issue": dict(issue),
                    "reason": "fine_grained_campp_turns_contradict_same_voice_claim",
                    "role_speakers": {
                        str(item["role"]): item["actual_speakers"][0]
                        for item in mentioned
                    },
                }
            )
            continue
        kept.append(issue)
    return tuple(kept), tuple(vetoed)


class AuralisAgent:
    """Run every audio evidence tool and then ask one judge to verify findings."""

    def __init__(
        self,
        *,
        probe_video: Callable[[Path], Mapping[str, Any]] = probe_video,
        extract_audio: Callable[[Path, Path], Any] = extract_audio_wav,
        transcribe_speech: Callable[[Path], Any] | None = None,
        transcribe_speech_with_prompt: PromptAwareTranscriber | None = None,
        extract_subtitles: Callable[[Path], Any] | None = None,
        align_speech_subtitles: Callable[[Any, Any], Any] = (
            check_speech_subtitle_alignment
        ),
        score_prompt_candidates: PromptCandidateScorer | None = None,
        judge: Judge | None = None,
        local_only: bool = False,
    ) -> None:
        if judge is None:
            if not local_only:
                raise ValueError(
                    "AuralisAgent 必须提供 judge；仅提取本地证据时显式设置 "
                    "local_only=True。"
                )
            judge = _no_judge
        if transcribe_speech is None and transcribe_speech_with_prompt is None:
            from agents.auralis.constrained_asr import (
                evaluate_prompt_constrained_asr,
            )
            from tools.speech_transcription.backends.sensevoice import (
                SenseVoiceBackend,
            )

            asr_backend = SenseVoiceBackend()
            transcribe_speech_with_prompt = lambda path, prompt: (
                asr_backend.transcribe(path, user_prompt=prompt)
            )
            if score_prompt_candidates is None:
                score_prompt_candidates = lambda path, prompt, transcript: (
                    evaluate_prompt_constrained_asr(
                        path,
                        prompt,
                        transcript,
                        scorer=asr_backend.score_candidates,
                    )
                )
        if extract_subtitles is None:
            from tools.subtitle_extraction.backends.rapidocr import (
                RapidOCRBackend,
            )

            ocr_backend = RapidOCRBackend()
            extract_subtitles = lambda path: run_subtitle_extraction(
                path,
                backend=ocr_backend,
            )
        self._probe_video = probe_video
        self._extract_audio = extract_audio
        self._transcribe_speech = transcribe_speech
        self._transcribe_speech_with_prompt = transcribe_speech_with_prompt
        self._extract_subtitles = extract_subtitles
        self._align_speech_subtitles = align_speech_subtitles
        self._score_prompt_candidates = score_prompt_candidates
        self._judge = judge

    def analyze(self, agent_input: AuralisInput) -> AuralisResult:
        metadata = self._probe_video(agent_input.video_path)
        if not bool(metadata.get("has_audio")):
            return AuralisResult(
                status="no_audio",
                diagnostics={"reason": "ffprobe did not detect an audio stream"},
            )

        with tempfile.TemporaryDirectory(prefix="auralis_") as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            extracted = self._extract_audio(agent_input.video_path, audio_path)
            if isinstance(extracted, Path):
                audio_path = extracted
            if self._transcribe_speech_with_prompt is not None:
                transcript = self._transcribe_speech_with_prompt(
                    audio_path,
                    agent_input.user_prompt,
                )
            elif self._transcribe_speech is not None:
                transcript = self._transcribe_speech(audio_path)
            else:  # pragma: no cover - constructor always configures one path.
                raise RuntimeError("Auralis ASR transcriber 未配置")
            constrained_asr: Mapping[str, Any] = {}
            if self._score_prompt_candidates is not None:
                try:
                    constrained_asr = self._score_prompt_candidates(
                        audio_path,
                        agent_input.user_prompt,
                        transcript,
                    )
                except Exception as exc:
                    # Preserve the existing ASR/OCR/Gemini path and expose a
                    # machine-readable failure instead of losing the row.
                    constrained_asr = {
                        "status": "scoring_failed",
                        "reason": "prompt_candidate_pipeline_exception",
                        "scoring_error": f"{type(exc).__name__}: {exc}",
                        "candidate_scores": [],
                    }
            subtitles = self._extract_subtitles(agent_input.video_path)
            judge_subtitles, rejected_ocr_singletons = (
                subtitle_evidence_for_judge(subtitles)
            )
            alignment = self._align_speech_subtitles(
                transcript,
                judge_subtitles,
            )
            evidence = AuralisEvidence(
                media_metadata=metadata,
                transcript=transcript,
                subtitles=subtitles,
                alignment=alignment,
                constrained_asr=constrained_asr,
            )
            judge_evidence = AuralisEvidence(
                media_metadata=metadata,
                transcript=transcript,
                subtitles=judge_subtitles,
                alignment=alignment,
                constrained_asr=constrained_asr,
            )
            judged_issues, vetoed_pronunciation_issues = filter_contradicted_judge_issues(
                constrained_asr,
                tuple(self._judge(agent_input, judge_evidence)),
            )
            judged_issues, vetoed_binding_issues = (
                filter_acoustically_contradicted_binding_issues(
                    transcript,
                    judged_issues,
                )
            )
            judged_issues, vetoed_ocr_issues = filter_unverified_ocr_judge_issues(
                judge_subtitles,
                alignment,
                rejected_ocr_singletons,
                judged_issues,
            )
            local_asr_issues = constrained_asr_issues(constrained_asr)
            local_alignment_issues = deterministic_alignment_issues(alignment)
            deterministic_issues = local_asr_issues + local_alignment_issues
            issues = deterministic_issues + judged_issues
        diagnostics: Mapping[str, Any] = {}
        diagnostics_payload: dict[str, Any] = {}
        if vetoed_pronunciation_issues:
            diagnostics_payload["constrained_asr_vetoed_judge_issues"] = (
                vetoed_pronunciation_issues
            )
        if vetoed_binding_issues:
            diagnostics_payload["campp_vetoed_binding_issues"] = (
                vetoed_binding_issues
            )
        if rejected_ocr_singletons:
            diagnostics_payload["ocr_unverified_singletons"] = [
                {
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "text": segment.text,
                    "bbox": list(segment.bbox),
                    "confidence": segment.confidence,
                    "source": segment.source,
                }
                for segment in rejected_ocr_singletons
            ]
        if vetoed_ocr_issues:
            diagnostics_payload["ocr_vetoed_judge_issues"] = vetoed_ocr_issues
        if diagnostics_payload:
            diagnostics = diagnostics_payload
        return AuralisResult(
            status="ok",
            issues=issues,
            deterministic_issues=deterministic_issues,
            evidence=evidence,
            diagnostics=diagnostics,
        )
