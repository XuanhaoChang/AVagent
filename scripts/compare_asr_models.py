#!/usr/bin/env python3
"""Compare native Faster-Whisper large-v3 with SenseVoice-Small.

This script intentionally runs both models in one isolated environment and
keeps raw model responses alongside a small normalized segment view.  It is a
comparison tool, not a replacement for the Auralis backend.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterable, Mapping


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Faster-Whisper large-v3 and SenseVoice-Small."
    )
    parser.add_argument("video", type=Path)
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="保存原始结果和归一化分段的 JSON 文件。",
    )
    parser.add_argument(
        "--large-v3-model",
        default="large-v3",
        help="Faster-Whisper 模型名称或本地路径。",
    )
    parser.add_argument(
        "--sensevoice-model",
        default="iic/SenseVoiceSmall",
        help="SenseVoice 模型名称或本地路径。",
    )
    parser.add_argument(
        "--sensevoice-replacement",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help=(
            "在 SenseVoice 自由识别文本中替换一次 OLD，并用同一份 CTC "
            "emission 比较原识别与替换候选；可重复指定。"
        ),
    )
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--language", default="zh")
    return parser.parse_args()


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return str(value)


def _native_large_v3(
    audio_path: Path,
    *,
    model_name: str,
    device: str,
    language: str,
) -> dict[str, Any]:
    from faster_whisper import WhisperModel

    started = time.monotonic()
    model = WhisperModel(
        model_name,
        device=device,
        compute_type="int8_float16" if device == "cuda" else "int8",
    )
    load_elapsed = time.monotonic() - started
    started = time.monotonic()
    segments, info = model.transcribe(
        str(audio_path),
        language=language,
        task="transcribe",
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
    )
    raw_segments = []
    for segment in segments:
        raw_segments.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": str(segment.text),
                "words": [
                    {
                        "start": float(word.start),
                        "end": float(word.end),
                        "word": str(word.word),
                        "probability": (
                            float(word.probability)
                            if word.probability is not None
                            else None
                        ),
                    }
                    for word in (segment.words or ())
                ],
            }
        )
    return {
        "model": model_name,
        "backend": "faster-whisper",
        "device": device,
        "language": str(getattr(info, "language", "") or ""),
        "load_elapsed_sec": round(load_elapsed, 3),
        "inference_elapsed_sec": round(time.monotonic() - started, 3),
        "segments": raw_segments,
        "raw": {
            "language_probability": getattr(info, "language_probability", None),
            "duration": getattr(info, "duration", None),
        },
    }


def _sensevoice_segments(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: Iterable[Any] = result.get("sentence_info") or ()
    normalized = []
    for item in candidates:
        if not isinstance(item, Mapping):
            continue
        text = item.get("text", item.get("sentence", ""))
        if text is None:
            text = ""
        start = item.get("start")
        end = item.get("end")
        if start is None or end is None:
            continue
        normalized.append(
            {
                "start_sec": float(start) / 1000.0,
                "end_sec": float(end) / 1000.0,
                "text": str(text),
                "speaker": item.get("spk"),
            }
        )
    if normalized:
        return normalized

    word_segments = _sensevoice_word_timestamps(result)
    if word_segments:
        return [
            {**item, "speaker": None}
            for item in word_segments
        ]

    timestamps = result.get("timestamp") or ()
    text = str(result.get("text", ""))
    if timestamps and text:
        return [
            {
                "start_sec": float(pair[0]) / 1000.0,
                "end_sec": float(pair[1]) / 1000.0,
                "text": text,
                "speaker": None,
            }
            for pair in timestamps
            if isinstance(pair, (list, tuple)) and len(pair) >= 2
        ]
    return []


def _sensevoice_word_timestamps(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize SenseVoice's CTC timestamps without treating them as sentences."""
    timestamps = result.get("timestamp") or ()
    words = result.get("words") or ()
    if not isinstance(timestamps, (list, tuple)) or not isinstance(words, (list, tuple)):
        return []
    if len(timestamps) != len(words):
        return []
    normalized = []
    for word, pair in zip(words, timestamps):
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            continue
        normalized.append(
            {
                "start_sec": float(pair[0]) / 1000.0,
                "end_sec": float(pair[1]) / 1000.0,
                "text": str(word),
            }
        )
    return normalized


def _sensevoice_ctc_confidence(
    model: Any,
    result: Mapping[str, Any],
    raw_logits: Any,
) -> list[dict[str, Any]]:
    """Return frame-aligned CTC confidence for SenseVoice tokens.

    FunASR exposes SenseVoice timestamps but not token probabilities.  The
    model is CTC-based, so we capture the output of its CTC projection and
    score the same forced-alignment path used for the timestamps.  This is a
    model confidence signal, not a calibrated probability of correctness.
    """
    if raw_logits is None:
        return []
    try:
        import torch
        from funasr.models.sense_voice.utils.ctc_alignment import ctc_forced_align

        tokenizer = model.kwargs.get("tokenizer")
        text = str(result.get("text") or "")
        if tokenizer is None or not text:
            return []
        tokens = tokenizer.text2tokens(text)[4:]
        token_back_to_id = tokenizer.tokens2ids(tokens)
        token_ids: list[int] = []
        for token_ids_for_token in token_back_to_id:
            if token_ids_for_token:
                token_ids.extend(token_ids_for_token)
            else:
                token_ids.append(124)
        if not token_ids:
            return []

        logits = raw_logits[0] if raw_logits.ndim == 3 else raw_logits
        log_probs = torch.log_softmax(logits, dim=-1)[4:]
        if log_probs.numel() == 0:
            return []
        target = torch.tensor(token_ids, dtype=torch.long, device=log_probs.device)
        aligned = ctc_forced_align(
            log_probs.unsqueeze(0).float(),
            target.unsqueeze(0),
            torch.tensor([log_probs.size(0)], dtype=torch.long, device=log_probs.device),
            torch.tensor([len(token_ids)], dtype=torch.long, device=log_probs.device),
            ignore_id=-1,
        )[0]

        records: list[dict[str, Any]] = []
        token_index = 0
        frame_start = 0
        for frame_end in range(1, aligned.numel() + 1):
            if frame_end < aligned.numel() and aligned[frame_end] == aligned[frame_start]:
                continue
            label = int(aligned[frame_start].item())
            if label != 0 and token_index < len(tokens):
                frame_slice = log_probs[frame_start:frame_end]
                token_log_prob = frame_slice[:, label]
                top2 = torch.topk(frame_slice, k=min(2, frame_slice.size(-1)), dim=-1).values
                margin = (
                    float((top2[:, 0] - top2[:, 1]).mean().item())
                    if top2.size(-1) > 1
                    else None
                )
                start_ms = max(0.0, (frame_start * 60.0) - 30.0)
                end_ms = min(
                    (frame_end * 60.0) - 30.0,
                    (log_probs.size(0) * 60.0) - 30.0,
                )
                records.append(
                    {
                        "token": str(tokens[token_index]),
                        "start_sec": round(start_ms / 1000.0, 4),
                        "end_sec": round(max(start_ms, end_ms) / 1000.0, 4),
                        "confidence": round(float(token_log_prob.mean().exp().item()), 6),
                        "log_probability": round(float(token_log_prob.mean().item()), 6),
                        "top2_margin": round(margin, 6) if margin is not None else None,
                        "frame_count": int(frame_end - frame_start),
                    }
                )
                token_index += 1
            frame_start = frame_end
        return records
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def _sensevoice_candidate_scores(
    model: Any,
    result: Mapping[str, Any],
    raw_logits: Any,
    replacements: Iterable[str],
) -> list[dict[str, Any]]:
    """Compare CTC sequence likelihoods after constrained text replacement."""
    replacement_specs = list(replacements)
    if raw_logits is None or not replacement_specs:
        return []
    try:
        import torch
        import torch.nn.functional as functional

        tokenizer = model.kwargs.get("tokenizer")
        raw_text = str(result.get("text") or "")
        if tokenizer is None or not raw_text:
            return []
        raw_tokens = tokenizer.text2tokens(raw_text)
        base_text = tokenizer.tokens2text(raw_tokens[4:])
        logits = raw_logits[0] if raw_logits.ndim == 3 else raw_logits
        log_probs = torch.log_softmax(logits, dim=-1)[4:].float()

        def encode_text(text: str) -> list[int]:
            pieces = tokenizer.text2tokens(text)
            encoded = tokenizer.tokens2ids(pieces)
            token_ids: list[int] = []
            for item in encoded:
                if isinstance(item, (list, tuple)):
                    token_ids.extend(int(value) for value in item)
                else:
                    token_ids.append(int(item))
            return token_ids

        comparisons: list[dict[str, Any]] = []
        for spec in replacement_specs:
            if "=" not in spec:
                comparisons.append({"replacement": spec, "error": "格式必须为 OLD=NEW"})
                continue
            old, new = spec.split("=", 1)
            if not old or old not in base_text:
                comparisons.append(
                    {
                        "replacement": spec,
                        "error": f"自由识别文本中不存在待替换片段：{old}",
                    }
                )
                continue
            candidate_text = base_text.replace(old, new, 1)
            variants = [
                ("recognized", base_text),
                ("replacement", candidate_text),
            ]
            scored: list[dict[str, Any]] = []
            for label, text in variants:
                token_ids = encode_text(text)
                target = torch.tensor(token_ids, dtype=torch.long)
                loss = functional.ctc_loss(
                    log_probs.unsqueeze(1),
                    target,
                    torch.tensor([log_probs.size(0)], dtype=torch.long),
                    torch.tensor([len(token_ids)], dtype=torch.long),
                    blank=0,
                    reduction="sum",
                    zero_infinity=False,
                )
                log_likelihood = -float(loss.item())
                scored.append(
                    {
                        "label": label,
                        "text": text,
                        "token_count": len(token_ids),
                        "ctc_log_likelihood": round(log_likelihood, 6),
                        "normalized_log_likelihood": round(
                            log_likelihood / max(1, len(token_ids)),
                            6,
                        ),
                    }
                )
            best_score = max(item["ctc_log_likelihood"] for item in scored)
            weights = [
                float(torch.exp(torch.tensor(item["ctc_log_likelihood"] - best_score)).item())
                for item in scored
            ]
            denominator = sum(weights)
            for item, weight in zip(scored, weights):
                item["relative_probability"] = round(weight / denominator, 6)
                item["delta_from_best"] = round(
                    item["ctc_log_likelihood"] - best_score,
                    6,
                )
            comparisons.append(
                {
                    "replacement": spec,
                    "old": old,
                    "new": new,
                    "candidates": scored,
                }
            )
        return comparisons
    except Exception as exc:
        return [{"error": f"{type(exc).__name__}: {exc}"}]


def _sensevoice(
    audio_path: Path,
    *,
    model_name: str,
    device: str,
    language: str,
    replacements: Iterable[str] = (),
) -> dict[str, Any]:
    from funasr import AutoModel
    from funasr.utils.postprocess_utils import rich_transcription_postprocess

    started = time.monotonic()
    # Run the comparison without VAD so one CTC emission tensor corresponds
    # to the one returned transcript.  This is only for the confidence probe;
    # the production Auralis worker still uses VAD and CAM++.
    model = AutoModel(
        model=model_name,
        device="cuda:0" if device == "cuda" else "cpu",
        disable_update=True,
    )
    captured_logits: list[Any] = []
    ctc_projection = getattr(getattr(model, "model", None), "ctc", None)
    ctc_projection = getattr(ctc_projection, "ctc_lo", None)
    hook = None
    if ctc_projection is not None:
        def capture_logits(_module: Any, _inputs: Any, output: Any) -> None:
            if not captured_logits:
                captured_logits.append(output.detach().cpu())

        hook = ctc_projection.register_forward_hook(capture_logits)
    load_elapsed = time.monotonic() - started
    started = time.monotonic()
    try:
        result = model.generate(
            input=str(audio_path),
            cache={},
            language=language,
            use_itn=True,
            batch_size_s=60,
            output_timestamp=True,
        )
    finally:
        if hook is not None:
            hook.remove()
    raw = result[0] if result else {}
    if not isinstance(raw, Mapping):
        raw = {"result": raw}
    raw_copy = _jsonable(raw)
    raw_text = str(raw.get("text", ""))
    return {
        "model": model_name,
        "backend": "funasr-sensevoice",
        "device": device,
        "language": language,
        "load_elapsed_sec": round(load_elapsed, 3),
        "inference_elapsed_sec": round(time.monotonic() - started, 3),
        "text": rich_transcription_postprocess(raw_text),
        "segments": _sensevoice_segments(raw),
        "word_timestamps": _sensevoice_word_timestamps(raw),
        "confidence_words": _sensevoice_ctc_confidence(
            model,
            raw,
            captured_logits[0] if captured_logits else None,
        ),
        "candidate_scores": _sensevoice_candidate_scores(
            model,
            raw,
            captured_logits[0] if captured_logits else None,
            replacements,
        ),
        "raw": raw_copy,
    }


def main() -> int:
    args = parse_args()
    video = args.video.resolve()
    if not video.is_file():
        raise FileNotFoundError(video)

    from tools.media.ffmpeg import extract_audio_wav

    with tempfile.TemporaryDirectory(prefix="asr_compare_") as temp_dir:
        audio_path = extract_audio_wav(video, Path(temp_dir) / "audio.wav")
        large_v3 = _native_large_v3(
            audio_path,
            model_name=args.large_v3_model,
            device=args.device,
            language=args.language,
        )
        sensevoice = _sensevoice(
            audio_path,
        model_name=args.sensevoice_model,
        device=args.device,
        language=args.language,
        replacements=args.sensevoice_replacement,
    )

    payload = {
        "video": str(video),
        "audio_format": {"sample_rate": 16000, "channels": 1},
        "large_v3": large_v3,
        "sensevoice_small": sensevoice,
    }
    output = args.output_json.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    print(f"saved: {output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
