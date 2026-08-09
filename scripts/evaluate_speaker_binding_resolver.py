#!/usr/bin/env python3
"""Evaluate the deterministic speaker-binding resolver on saved evidence."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agents.auralis.speaker_binding_resolver import resolve_speaker_binding
from agents.auralis.speaker_plan import extract_prompt_speech_plan
from agents.auralis.speaker_voiceprint import (
    build_role_voiceprint_clips,
    evaluate_prompt_voiceprints,
)
from tools.media.ffmpeg import extract_audio_wav
from tools.speech_transcription.backends.sensevoice import (
    SenseVoiceBackend,
    _prompt_turn_speaker_alignment,
)
from tools.speech_transcription.schemas import SpeechSegment, SpeechTranscript
from tools.speech_transcription.speaker_turns import sentence_info_to_segments


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _segments(raw_transcript: dict[str, Any]) -> tuple[SpeechSegment, ...]:
    metadata = raw_transcript.get("metadata", {})
    clustering = metadata.get("clustering", {})
    raw_label_map = clustering.get("raw_to_anonymous_label", {})
    label_map = {
        int(key) if str(key).lstrip("-").isdigit() else str(key): value
        for key, value in raw_label_map.items()
    }
    rebuilt = sentence_info_to_segments(
        metadata.get("raw_sentence_info", ()),
        metadata.get("speaker_diarization", {}).get("speaker_turns", ()),
        label_map,
    )
    raw_segments = rebuilt or raw_transcript.get("segments", ())
    return tuple(
        SpeechSegment(
            start_sec=float(item["start_sec"]),
            end_sec=float(item["end_sec"]),
            text=str(item.get("text") or ""),
            confidence=str(item.get("confidence") or "medium"),
            speaker=item.get("speaker"),
        )
        for item in raw_segments
        if isinstance(item, dict)
        and item.get("start_sec") is not None
        and item.get("end_sec") is not None
    )


def evaluate_sample(
    sample_root: Path,
    *,
    voiceprint_backend: SenseVoiceBackend | None = None,
) -> dict[str, Any]:
    input_data = _read_json(sample_root / "input.json")
    asr_data = _read_json(sample_root / "asr.json")
    gt_data = _read_json(sample_root / "gt.json")
    transcript = asr_data.get("transcript", {})
    metadata = transcript.get("metadata", {})
    segments = _segments(transcript)
    plan = extract_prompt_speech_plan(str(input_data.get("user_prompt") or ""))
    alignments = _prompt_turn_speaker_alignment(plan, segments)
    existing_binding = metadata.get("speaker_binding_evidence", {})
    speaker_turns = metadata.get("speaker_diarization", {}).get(
        "speaker_turns", ()
    )
    observed_speakers = {
        segment.speaker for segment in segments if segment.speaker is not None
    }
    binding_status = (
        "fine_grained_turns"
        if speaker_turns
        else str(existing_binding.get("status") or "")
        if isinstance(existing_binding, dict)
        else ""
    )
    if binding_status not in {
        "fine_grained_turns",
        "sentence_labels_only",
        "unavailable",
    }:
        binding_status = (
            "sentence_labels_only" if observed_speakers else "unavailable"
        )
    replay_metadata = {
        **metadata,
        "prompt_speech_plan": plan,
        "speaker_binding_evidence": {
            **(
                existing_binding
                if isinstance(existing_binding, dict)
                else {}
            ),
            "status": binding_status,
            "prompt_scope": plan.get("scope"),
            "prompt_turn_alignment": alignments,
        },
    }
    replay_transcript = SpeechTranscript(
        language=str(transcript.get("language") or "auto"),
        segments=segments,
        backend=str(transcript.get("backend") or "saved-replay"),
        model=str(transcript.get("model") or ""),
        device=str(transcript.get("device") or ""),
        metadata=replay_metadata,
    )
    if voiceprint_backend is None:
        voiceprint = {
            "version": 1,
            "status": "not_evaluable",
            "reason": "offline_voiceprint_scoring_disabled",
            "clips": [],
            "pairs": [],
        }
    else:
        eligible_roles = {
            str(item["role"])
            for item in build_role_voiceprint_clips(replay_transcript)
            if item["eligible"]
        }
        if (
            len(eligible_roles) < 2
            or str(metadata.get("speech_evidence_status") or "")
            in {"bgm_only", "speech_with_bgm"}
        ):
            voiceprint = evaluate_prompt_voiceprints(
                Path("unused.wav"),
                replay_transcript,
                scorer=lambda *_args: {},
            )
        else:
            with tempfile.TemporaryDirectory(
                prefix="speaker_voiceprint_replay_"
            ) as temp_dir:
                audio_path = extract_audio_wav(
                    sample_root / "video.mp4",
                    Path(temp_dir) / "audio.wav",
                )
                voiceprint = evaluate_prompt_voiceprints(
                    audio_path,
                    replay_transcript,
                    scorer=voiceprint_backend.score_speaker_segments,
                )
    result = resolve_speaker_binding(
        plan,
        alignments,
        binding_status=binding_status,
        speech_evidence_status=str(
            metadata.get("speech_evidence_status") or "speech_present"
        ),
        clustering=metadata.get("clustering", {}),
        voiceprint_evidence=voiceprint,
    )
    prompt = str(input_data.get("user_prompt") or "")
    gt_binding_expected = any(
        isinstance(issue, dict)
        and "台词归属"
        in (
            str(issue.get("问题类型") or "")
            + str(issue.get("问题说明") or "")
        )
        for issue in gt_data
    )
    return {
        "sample_directory": sample_root.name,
        "sample_id": str(input_data.get("序号") or ""),
        "feedback": str(input_data.get("用户反馈") or ""),
        "source_asr_backend": str(transcript.get("backend") or ""),
        "source_asr_device": str(transcript.get("device") or ""),
        "gt_binding_expected": gt_binding_expected,
        "extraction": {
            "scope": plan.get("scope"),
            "roles": plan.get("expected_speaking_roles", []),
            "turn_count": len(plan.get("turns", ())),
            "source_span_valid": all(
                prompt[int(turn["prompt_start"]) : int(turn["prompt_end"])]
                == str(turn.get("prompt_source_text") or "")
                for turn in plan.get("turns", ())
            ),
            "anchored_turn_count": sum(
                item.get("status") == "anchored" for item in alignments
            ),
        },
        "voiceprint": voiceprint,
        "resolver": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--score-voiceprints", action="store_true")
    parser.add_argument(
        "--voiceprint-device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    args = parser.parse_args()

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    voiceprint_backend = (
        SenseVoiceBackend(device=args.voiceprint_device)
        if args.score_voiceprints
        else None
    )
    try:
        for sample_root in sorted(args.samples_root.glob("sample_*")):
            if not sample_root.is_dir():
                continue
            try:
                records.append(
                    evaluate_sample(
                        sample_root,
                        voiceprint_backend=voiceprint_backend,
                    )
                )
            except Exception as exc:
                errors.append(
                    {
                        "sample_directory": sample_root.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    finally:
        if voiceprint_backend is not None:
            voiceprint_backend.close()
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False) + "\n" for record in records
        ),
        encoding="utf-8",
    )
    decisions = Counter(
        str(record["resolver"].get("decision") or "unknown")
        for record in records
    )
    supported_samples = {
        record["sample_directory"]
        for record in records
        if record["resolver"].get("decision") == "supported"
    }
    directional_supported_samples = {
        record["sample_directory"]
        for record in records
        if record["resolver"].get("directional_conflicts")
    }
    shared_voice_supported_samples = {
        record["sample_directory"]
        for record in records
        if record["resolver"].get("shared_voice_conflicts")
    }
    gt_binding_samples = {
        record["sample_directory"]
        for record in records
        if record["gt_binding_expected"]
    }
    true_positives = directional_supported_samples & gt_binding_samples
    false_positives = directional_supported_samples - gt_binding_samples
    false_negatives = gt_binding_samples - directional_supported_samples
    scope_counts = Counter(
        str(record["extraction"].get("scope") or "unknown")
        for record in records
    )
    reason_counts = Counter(
        str(record["resolver"].get("reason") or "unknown")
        for record in records
    )
    summary = {
        "sample_count": len(records),
        "error_count": len(errors),
        "decision_counts": dict(sorted(decisions.items())),
        "issue_count": sum(
            len(record["resolver"].get("issues", ())) for record in records
        ),
        "supported_samples": sorted(supported_samples),
        "directional_supported_samples": sorted(directional_supported_samples),
        "shared_voice_supported_samples": sorted(shared_voice_supported_samples),
        "voiceprint": {
            "enabled": args.score_voiceprints,
            "device": args.voiceprint_device if args.score_voiceprints else "disabled",
            "status_counts": dict(
                sorted(
                    Counter(
                        str(record["voiceprint"].get("status") or "unknown")
                        for record in records
                    ).items()
                )
            ),
            "pair_decision_counts": dict(
                sorted(
                    Counter(
                        str(pair.get("decision") or "unknown")
                        for record in records
                        for pair in record["voiceprint"].get("pairs", ())
                        if isinstance(pair, dict)
                    ).items()
                )
            ),
        },
        "evidence_coverage": {
            "scope_counts": dict(sorted(scope_counts.items())),
            "prompts_with_dialogue": sum(
                record["extraction"]["turn_count"] > 0 for record in records
            ),
            "multi_role_prompts": sum(
                len(record["extraction"]["roles"]) >= 2 for record in records
            ),
            "samples_with_anchored_turns": sum(
                record["extraction"]["anchored_turn_count"] > 0
                for record in records
            ),
            "samples_with_two_or_more_anchors": sum(
                record["extraction"]["anchored_turn_count"] >= 2
                for record in records
            ),
            "samples_with_split_or_shared_candidates": sum(
                bool(record["resolver"].get("split_role_candidates"))
                or bool(record["resolver"].get("shared_speaker_candidates"))
                for record in records
            ),
            "reason_counts": dict(sorted(reason_counts.items())),
            "note": (
                "Saved ASR evidence predates the latest BGM lexical-speech fix; "
                "bgm_only rows underestimate current evaluability."
            ),
        },
        "gt_audit": {
            "scope": "GT problem type or description explicitly contains 台词归属",
            "gt_binding_samples": sorted(gt_binding_samples),
            "true_positives": sorted(true_positives),
            "false_positives": sorted(false_positives),
            "false_negatives": sorted(false_negatives),
            "precision": (
                len(true_positives) / len(directional_supported_samples)
                if directional_supported_samples
                else None
            ),
            "recall": (
                len(true_positives) / len(gt_binding_samples)
                if gt_binding_samples
                else None
            ),
        },
        "invalid_source_span_samples": [
            record["sample_directory"]
            for record in records
            if not record["extraction"]["source_span_valid"]
        ],
        "errors": errors,
    }
    args.summary_json.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
