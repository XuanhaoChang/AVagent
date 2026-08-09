"""SenseVoice-Small + CAM++ backend running in the isolated avbench env."""

from __future__ import annotations

import ast
from difflib import SequenceMatcher
import json
import os
from pathlib import Path
import re
import subprocess
from threading import Lock
from typing import Any, Mapping, Sequence

from agents.auralis.speaker_plan import extract_prompt_speech_plan

from ..schemas import SpeechSegment, SpeechTranscript


REPO_ROOT = Path(__file__).resolve().parents[3]
_EVENT_MARKERS = ("🎼", "😊", "😔", "😡", "😐")
_SENSEVOICE_EVENT = re.compile(r"<\|(?P<event>Speech|BGM)\|>", re.IGNORECASE)
_CJK_CHARACTER = re.compile(r"[\u3400-\u9fff\u3040-\u30ff\uac00-\ud7af]")
_LATIN_WORD = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")
_BGM_PLACEHOLDER_WORDS = {
    "a",
    "ah",
    "an",
    "eh",
    "hmm",
    "oh",
    "the",
    "uh",
    "um",
    "yeah",
}


def _flatten_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(
            item
            for item in (_flatten_text(part).strip() for part in value)
            if item
        )
    text = str(value)
    stripped = text.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return _flatten_text(parsed)
    return text


def _clean_text(value: Any) -> str:
    text = _flatten_text(value)
    while "<|" in text and "|>" in text:
        start = text.find("<|")
        end = text.find("|>", start)
        if end < 0:
            break
        text = text[:start] + text[end + 2 :]
    for marker in _EVENT_MARKERS:
        text = text.replace(marker, "")
    return " ".join(text.split()).strip(" ，。！？、")


def _has_substantial_lexical_speech(
    raw_text: Any,
    raw_segments: Sequence[Mapping[str, Any]],
) -> bool:
    """Distinguish real lexical output from common BGM hallucination tokens."""

    segment_text = " ".join(
        text
        for text in (
            _clean_text(item.get("text"))
            for item in raw_segments
            if isinstance(item, Mapping)
        )
        if text
    )
    text = segment_text or _clean_text(raw_text)
    cjk_characters = _CJK_CHARACTER.findall(text)
    if len(cjk_characters) >= 4:
        return True
    latin_words = [word.casefold() for word in _LATIN_WORD.findall(text)]
    informative_words = [
        word for word in latin_words if word not in _BGM_PLACEHOLDER_WORDS
    ]
    return len(informative_words) >= 3 and sum(map(len, informative_words)) >= 8


def _sensevoice_event_evidence(
    raw_text: Any,
    raw_segments: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[str], str]:
    """Classify SenseVoice rich tags before their text is normalized away.

    ``<|BGM|>`` means that music is present, not necessarily that speech is
    absent.  Keep substantial timestamped lexical output as speech-with-BGM,
    while suppressing short placeholders such as repeated ``The.``.
    """

    events = list(
        dict.fromkeys(
            match.group("event").casefold()
            for match in _SENSEVOICE_EVENT.finditer(str(raw_text or ""))
        )
    )
    if "speech" in events:
        return events, "speech_with_bgm" if "bgm" in events else "speech_present"
    if "bgm" in events and _has_substantial_lexical_speech(raw_text, raw_segments):
        return events, "speech_with_bgm"
    if "bgm" in events:
        return events, "bgm_only"
    return events, "unclassified"


def _prompt_turn_speaker_alignment(
    prompt_speech_plan: Mapping[str, Any],
    segments: Sequence[SpeechSegment],
) -> list[dict[str, Any]]:
    """Map prompt dialogue turns to observed anonymous speakers."""

    prompt_turns = [
        turn
        for turn in prompt_speech_plan.get("turns", ())
        if isinstance(turn, Mapping)
    ]
    normalized_dialogues = [
        "".join(
            character.lower()
            for character in str(turn.get("dialogue_text") or "")
            if character.isalnum()
        )
        for turn in prompt_turns
    ]
    text_assignments: dict[int, list[tuple[SpeechSegment, float, float]]] = {
        index: [] for index in range(len(prompt_turns))
    }
    for segment in segments:
        observed = "".join(
            character.lower()
            for character in segment.text
            if character.isalnum()
        )
        if len(observed) < 2 or segment.speaker is None:
            continue
        scores = []
        observed_precisions = []
        for expected in normalized_dialogues:
            if not expected:
                scores.append(0.0)
                observed_precisions.append(0.0)
                continue
            matcher = SequenceMatcher(None, expected, observed)
            matched = sum(block.size for block in matcher.get_matching_blocks())
            scores.append(
                max(
                    matcher.ratio(),
                    matched / max(1, min(len(expected), len(observed))),
                )
            )
            observed_precisions.append(matched / max(1, len(observed)))
        if not scores:
            continue
        winner = max(range(len(scores)), key=scores.__getitem__)
        runner_up = max(
            (score for index, score in enumerate(scores) if index != winner),
            default=0.0,
        )
        if scores[winner] >= 0.55 and scores[winner] - runner_up >= 0.10:
            text_assignments[winner].append(
                (segment, scores[winner], observed_precisions[winner])
            )

    aligned: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(prompt_turns):
        start = turn.get("expected_start_sec")
        end = turn.get("expected_end_sec")
        has_time_window = (
            start is not None
            and end is not None
            and float(end) > float(start)
        )
        speaker_durations: dict[str | int, float] = {}
        matched_text: list[str] = []
        text_matches = text_assignments.get(turn_index, [])
        if text_matches:
            for segment, _score, _observed_precision in text_matches:
                matched_text.append(segment.text)
                speaker_durations[segment.speaker] = (
                    speaker_durations.get(segment.speaker, 0.0)
                    + segment.end_sec
                    - segment.start_sec
                )
            anchor_method = "dialogue_text_similarity"
        elif has_time_window:
            for segment in segments:
                overlap = min(float(end), segment.end_sec) - max(
                    float(start), segment.start_sec
                )
                if overlap <= 0:
                    continue
                matched_text.append(segment.text)
                if segment.speaker is not None:
                    speaker_durations[segment.speaker] = (
                        speaker_durations.get(segment.speaker, 0.0) + overlap
                    )
            anchor_method = "time_window_only"
        else:
            continue
        supported_speakers = [
            speaker
            for speaker, duration in sorted(
                speaker_durations.items(),
                key=lambda item: (-item[1], str(item[0])),
            )
            if duration >= 0.20
        ]
        alignment = {
            "role": str(turn.get("role") or ""),
            "dialogue_text": str(turn.get("dialogue_text") or ""),
            "actual_speakers": supported_speakers,
            "speaker_overlap_sec": {
                str(speaker): round(duration, 3)
                for speaker, duration in speaker_durations.items()
            },
            "observed_text": "".join(matched_text),
            "status": (
                "anchored"
                if supported_speakers and text_matches
                else "time_window_only"
                if supported_speakers
                else "unmatched"
            ),
            "anchor_method": anchor_method,
            "dialogue_match_scores": [
                round(score, 3)
                for _segment, score, _observed_precision in text_matches
            ],
            "dialogue_observed_precisions": [
                round(observed_precision, 3)
                for _segment, _score, observed_precision in text_matches
            ],
            "matched_segments": [
                {
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "speaker": segment.speaker,
                    "text": segment.text,
                    "dialogue_match_score": round(score, 3),
                    "dialogue_observed_precision": round(
                        observed_precision,
                        3,
                    ),
                }
                for segment, score, observed_precision in text_matches
            ],
        }
        if has_time_window:
            alignment["expected_start_sec"] = float(start)
            alignment["expected_end_sec"] = float(end)
        aligned.append(alignment)
    return aligned


def _speaker_binding_summary(
    prompt_speech_plan: Mapping[str, Any],
    segments: Sequence[SpeechSegment],
    alignments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Expose prompt-relative CAM++ checks without overriding clustering."""

    anchored = [
        item
        for item in alignments
        if item.get("status") == "anchored"
        and item.get("role")
        and item.get("actual_speakers")
    ]
    role_to_speakers: dict[str, list[str | int]] = {}
    for item in anchored:
        speakers = role_to_speakers.setdefault(str(item["role"]), [])
        for speaker in item.get("actual_speakers", ()):
            if speaker not in speakers:
                speakers.append(speaker)
    speaker_to_roles: dict[str, list[str]] = {}
    for role, speakers in role_to_speakers.items():
        for speaker in speakers:
            roles = speaker_to_roles.setdefault(str(speaker), [])
            if role not in roles:
                roles.append(role)
    split_roles = [
        {"role": role, "speakers": speakers}
        for role, speakers in role_to_speakers.items()
        if len(speakers) > 1
    ]
    shared_speakers = [
        {"speaker": speaker, "roles": roles}
        for speaker, roles in speaker_to_roles.items()
        if len(roles) > 1
    ]
    assigned_segment_keys = {
        (
            float(segment["start_sec"]),
            float(segment["end_sec"]),
            segment.get("speaker"),
        )
        for item in anchored
        for segment in item.get("matched_segments", ())
        if isinstance(segment, Mapping)
        and segment.get("start_sec") is not None
        and segment.get("end_sec") is not None
    }
    unassigned_segments = []
    for segment in segments:
        segment_key = (
            float(segment.start_sec),
            float(segment.end_sec),
            segment.speaker,
        )
        if segment_key in assigned_segment_keys:
            continue
        unassigned_segments.append(
            {
                "start_sec": segment.start_sec,
                "end_sec": segment.end_sec,
                "speaker": segment.speaker,
                "text": segment.text,
                "closed_script_candidate": (
                    prompt_speech_plan.get("scope") == "closed"
                ),
            }
        )
    return {
        "role_to_speakers": role_to_speakers,
        "speaker_to_roles": speaker_to_roles,
        "split_role_candidates": split_roles,
        "shared_speaker_candidates": shared_speakers,
        "unassigned_segments": unassigned_segments,
    }


class SenseVoiceBackend:
    """Persistent subprocess wrapper for FunASR SenseVoice and CAM++.

    The main Agent-D environment intentionally does not import PyTorch/FunASR.
    A long-lived worker in ``.conda-envs/avbench`` loads the models once and
    returns JSON so the ASR/OCR process remains lightweight and reproducible.
    """

    def __init__(
        self,
        *,
        model_name: str | None = None,
        device: str | None = None,
        language: str | None = None,
        python_executable: str | Path | None = None,
        use_campp: bool | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "AURALIS_SENSEVOICE_MODEL", "iic/SenseVoiceSmall"
        )
        self.device = device or os.getenv("AURALIS_ASR_DEVICE", "cuda")
        self.language = language or os.getenv("AURALIS_ASR_LANGUAGE", "auto")
        self.python_executable = Path(
            python_executable
            or os.getenv(
                "AURALIS_SENSEVOICE_PYTHON",
                str(REPO_ROOT / ".conda-envs" / "avbench" / "bin" / "python"),
            )
        )
        self.use_campp = (
            use_campp
            if use_campp is not None
            else os.getenv("AURALIS_USE_CAMPP", "1").strip().lower()
            in {"1", "true", "yes"}
        )
        self.campp_similarity_threshold = float(
            os.getenv("AURALIS_CAMPP_SIM_THRESHOLD", "0.78")
        )
        # Direct verification is deliberately stricter than the model card's
        # general speaker-verification operating point.  It is used to assert
        # a generated-video defect, so ambiguous pairs must abstain.
        self.campp_voiceprint_same_threshold = float(
            os.getenv("AURALIS_CAMPP_VERIFY_SAME_THRESHOLD", "0.55")
        )
        self.campp_voiceprint_different_threshold = float(
            os.getenv("AURALIS_CAMPP_VERIFY_DIFFERENT_THRESHOLD", "0.30")
        )
        self.fallback_reason = ""
        self._worker: subprocess.Popen[str] | None = None
        self._lock = Lock()

    def _start_worker(self) -> subprocess.Popen[str]:
        if not self.python_executable.is_file():
            raise RuntimeError(
                "SenseVoice Python 环境不存在："
                f"{self.python_executable}；请准备 .conda-envs/avbench。"
            )
        environment = os.environ.copy()
        environment.setdefault(
            "MODELSCOPE_CACHE", str(REPO_ROOT / "models" / "modelscope")
        )
        project_path = str(REPO_ROOT)
        existing_python_path = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            project_path
            if not existing_python_path
            else project_path + os.pathsep + existing_python_path
        )
        worker = subprocess.Popen(
            [str(self.python_executable), "-m", "tools.speech_transcription.sensevoice_worker"],
            cwd=str(REPO_ROOT),
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
        )
        self._worker = worker
        return worker

    def _close_worker(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is None:
            return
        try:
            if worker.stdin is not None:
                worker.stdin.close()
        except OSError:
            pass
        if worker.poll() is None:
            worker.terminate()
            try:
                worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.wait(timeout=5)

    def _worker_request(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        with self._lock:
            for attempt in range(2):
                worker = self._worker or self._start_worker()
                try:
                    if worker.stdin is None or worker.stdout is None:
                        raise RuntimeError("SenseVoice worker 管道未创建")
                    worker.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
                    worker.stdin.flush()
                    line = worker.stdout.readline()
                    if not line:
                        return_code = worker.poll()
                        raise RuntimeError(
                            "SenseVoice worker 提前退出，returncode="
                            f"{return_code}"
                        )
                    response = json.loads(line)
                    if not isinstance(response, Mapping):
                        raise RuntimeError("SenseVoice worker 返回格式错误")
                    if not bool(response.get("ok")):
                        raise RuntimeError(str(response.get("error") or "未知错误"))
                    return response
                except (BrokenPipeError, OSError, json.JSONDecodeError, RuntimeError):
                    self._close_worker()
                    if attempt == 1:
                        raise
        raise RuntimeError("SenseVoice worker 请求失败")

    def _request(self, audio_path: Path) -> Mapping[str, Any]:
        return self._worker_request(
            {
                "action": "transcribe",
                "audio_path": str(audio_path),
                "model_name": self.model_name,
                "device": self.device,
                "language": self.language,
                "use_campp": self.use_campp,
                "campp_similarity_threshold": self.campp_similarity_threshold,
            }
        )

    @staticmethod
    def _segments(response: Mapping[str, Any]) -> tuple[SpeechSegment, ...]:
        segments = []
        for item in response.get("segments", ()):
            if not isinstance(item, Mapping):
                continue
            text = _clean_text(item.get("text"))
            start = item.get("start_sec")
            end = item.get("end_sec")
            if not text or start is None or end is None or float(end) <= float(start):
                continue
            segments.append(
                SpeechSegment(
                    start_sec=float(start),
                    end_sec=float(end),
                    text=text,
                    confidence="medium",
                    speaker=item.get("speaker"),
                )
            )
        return tuple(segments)

    def transcribe(
        self,
        audio_path: Path,
        *,
        user_prompt: str = "",
    ) -> SpeechTranscript:
        if not audio_path.is_file():
            raise FileNotFoundError(f"待转写音频不存在：{audio_path}")
        response = self._request(audio_path)
        event_types, speech_evidence_status = _sensevoice_event_evidence(
            response.get("raw_text"),
            tuple(
                item
                for item in response.get("segments", ())
                if isinstance(item, Mapping)
            ),
        )
        # SenseVoice sometimes emits lexical-looking placeholders such as
        # "The." for instrumental music while explicitly tagging every event
        # as BGM. Such placeholders are not speech and must not enter CAM++
        # binding or ASR/OCR alignment.
        segments = (
            ()
            if speech_evidence_status == "bgm_only"
            else self._segments(response)
        )
        prompt_speech_plan = extract_prompt_speech_plan(user_prompt)
        speaker_segments = [
            {
                "start_sec": item.start_sec,
                "end_sec": item.end_sec,
                "speaker": item.speaker,
                "text": item.text,
            }
            for item in segments
        ]
        speaker_turns = [
            {
                "start_sec": float(item["start_sec"]),
                "end_sec": float(item["end_sec"]),
                "speaker": item.get("speaker"),
            }
            for item in response.get("speaker_turns", ())
            if isinstance(item, Mapping)
            and item.get("start_sec") is not None
            and item.get("end_sec") is not None
        ]
        if speech_evidence_status == "bgm_only":
            speaker_turns = []
        turn_speakers = {
            item["speaker"]
            for item in speaker_turns
            if item["speaker"] is not None
        }
        sentence_speakers = {
            item["speaker"]
            for item in speaker_segments
            if item["speaker"] is not None
        }
        clustering = dict(response.get("speaker_clustering") or {})
        if speech_evidence_status == "bgm_only":
            clustering["suppressed_bgm_embedding_count"] = int(
                clustering.get("embedding_count", 0)
            )
            clustering.update(
                {
                    "embedding_count": 0,
                    "embedding_labels": [],
                    "embedding_cluster_count": 0,
                    "speaker_turn_count": 0,
                    "turn_speaker_count": 0,
                    "sentence_speaker_count": 0,
                    "granularity_conflict": False,
                    "raw_to_anonymous_label": {},
                }
            )
        binding_status = (
            "fine_grained_turns"
            if speaker_turns
            else "sentence_labels_only"
            if sentence_speakers
            else "unavailable"
        )
        prompt_turn_alignment = _prompt_turn_speaker_alignment(
            prompt_speech_plan,
            segments,
        )
        binding_summary = _speaker_binding_summary(
            prompt_speech_plan,
            segments,
            prompt_turn_alignment,
        )
        return SpeechTranscript(
            language=str(response.get("language") or self.language),
            segments=segments,
            backend="funasr-sensevoice-campp" if self.use_campp else "funasr-sensevoice",
            model=(
                f"{self.model_name}+cam++"
                if self.use_campp
                else self.model_name
            ),
            device=f"{self.device}/funasr",
            metadata={
                "speaker_diarization": {
                    "backend": "CAM++" if self.use_campp else "disabled",
                    "speaker_count": len(turn_speakers or sentence_speakers),
                    "turn_speaker_count": len(turn_speakers),
                    "sentence_speaker_count": len(sentence_speakers),
                    "speaker_turns": speaker_turns,
                    "segments": speaker_segments,
                },
                "prompt_speech_plan": prompt_speech_plan,
                "speaker_binding_evidence": {
                    "status": binding_status,
                    "prompt_scope": prompt_speech_plan["scope"],
                    "expected_speaking_roles": prompt_speech_plan[
                        "expected_speaking_roles"
                    ],
                    "expected_speaker_count": prompt_speech_plan[
                        "expected_speaker_count"
                    ],
                    "actual_acoustic_speaker_count": len(
                        turn_speakers or sentence_speakers
                    ),
                    "hard_binding_requires_prompt_turn_anchor": True,
                    "prompt_count_is_not_cluster_ground_truth": True,
                    "prompt_turn_alignment": prompt_turn_alignment,
                    **binding_summary,
                },
                "raw_text": str(response.get("raw_text") or ""),
                "audio_event_types": event_types,
                "speech_evidence_status": speech_evidence_status,
                "raw_sentence_info": response.get("raw_sentence_info") or [],
                "clustering": clustering,
            },
        )

    def score_candidates(
        self,
        audio_path: Path,
        candidates: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Score prompt-derived observed/expected strings on local audio clips."""

        if not audio_path.is_file():
            raise FileNotFoundError(f"待评分音频不存在：{audio_path}")
        score_requests = []
        for item in candidates:
            score_requests.append(
                {
                    "candidate_id": str(item.get("candidate_id") or ""),
                    "start_sec": float(item["start_sec"]),
                    "end_sec": float(item["end_sec"]),
                    "observed_text": str(item["observed_text"]),
                    "expected_text": str(item["expected_text"]),
                }
            )
        return self._worker_request(
            {
                "action": "score_candidates",
                "audio_path": str(audio_path),
                "model_name": self.model_name,
                "device": self.device,
                "language": self.language,
                "use_campp": self.use_campp,
                "campp_similarity_threshold": self.campp_similarity_threshold,
                "candidates": score_requests,
            }
        )

    def score_speaker_segments(
        self,
        audio_path: Path,
        clips: Sequence[Mapping[str, Any]],
    ) -> Mapping[str, Any]:
        """Directly compare prompt-anchored clips with the local CAM++ model."""

        if not audio_path.is_file():
            raise FileNotFoundError(f"待评分音频不存在：{audio_path}")
        requests = [
            {
                "clip_id": str(item.get("clip_id") or ""),
                "start_sec": float(item["start_sec"]),
                "end_sec": float(item["end_sec"]),
            }
            for item in clips
        ]
        return self._worker_request(
            {
                "action": "score_speaker_segments",
                "audio_path": str(audio_path),
                "model_name": self.model_name,
                "device": self.device,
                "language": self.language,
                "use_campp": self.use_campp,
                "campp_similarity_threshold": self.campp_similarity_threshold,
                "same_speaker_threshold": (
                    self.campp_voiceprint_same_threshold
                ),
                "different_speaker_threshold": (
                    self.campp_voiceprint_different_threshold
                ),
                "clips": requests,
            }
        )

    def close(self) -> None:
        with self._lock:
            self._close_worker()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
