"""Line-oriented FunASR worker used by the Agent-D SenseVoice backend."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import traceback
from typing import Any, Mapping

from tools.speech_transcription.speaker_turns import (
    normalize_speaker_turns,
    sentence_info_to_segments,
)


# ModelScope/FunASR may print initialization logs to stdout. Keep the original
# stream for the JSON protocol and redirect all library chatter to stderr.
_PROTOCOL_STDOUT = sys.stdout
sys.stdout = sys.stderr
_MODEL_CACHE: dict[tuple[str, str, bool, float], Any] = {}


class _SmallSampleCamppCluster:
    """Short-clip CAM++ clustering with a conservative fallback.

    FunASR 1.3.29's ``ClusterBackend`` intentionally returns one speaker when
    fewer than 20 embeddings are available.  That is a useful conservative
    default for its general diarization path, but it suppresses CAM++ evidence
    for the short caption clips handled by Agent-D.  Reuse FunASR's spectral
    clustering and cosine merge threshold without that short-clip shortcut;
    if the tiny affinity matrix cannot be clustered, fall back to one speaker.
    """

    def __init__(self, merge_threshold: float = 0.78) -> None:
        self.merge_threshold = merge_threshold
        self.last_labels: list[int] = []

    def __call__(self, embeddings: Any, oracle_num: int | None = None):
        import numpy as np

        if hasattr(embeddings, "detach"):
            values = embeddings.detach().cpu().numpy()
        else:
            values = np.asarray(embeddings)
        values = np.asarray(values, dtype="float32")
        if values.ndim != 2:
            raise ValueError("CAM++ embeddings must have shape [N, C]")
        count = values.shape[0]
        if count == 0:
            self.last_labels = []
            return np.zeros(0, dtype="int")
        if count == 1:
            self.last_labels = [0]
            return np.zeros(1, dtype="int")

        # Use FunASR's normal spectral clustering algorithm without its
        # ``N < 20 -> one speaker`` shortcut.  Spectral clustering is more
        # suitable than pairwise connected components here: a short clip can
        # contain several imperfectly separated CAM++ windows, and direct
        # threshold edges otherwise create one new speaker per window.
        from funasr.models.campplus.cluster_backend import SpectralCluster

        max_num_spks = min(8, count - 1)
        try:
            labels = SpectralCluster(
                min_num_spks=1,
                max_num_spks=max_num_spks,
            )(values, oracle_num=oracle_num)
            labels = np.asarray(labels, dtype="int")
            if oracle_num is None and labels.size:
                from funasr.models.campplus.cluster_backend import ClusterBackend

                labels = ClusterBackend(merge_thr=self.merge_threshold).merge_by_cos(
                    labels, values, self.merge_threshold
                )
        except Exception:
            # A conservative fallback keeps the evidence usable if a future
            # sklearn/scipy version rejects a tiny affinity matrix.
            labels = np.zeros(count, dtype="int")
        label_by_root: dict[int, int] = {}
        labels = [
            label_by_root.setdefault(int(label), len(label_by_root))
            for label in labels
        ]
        self.last_labels = [int(label) for label in labels]
        return np.asarray(labels, dtype="int")


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return str(value)


def _flatten_text(value: Any) -> str:
    """Flatten FunASR list-valued text without leaking Python repr syntax."""

    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(
            item
            for item in (_flatten_text(part).strip() for part in value)
            if item
        )
    return str(value)


def _load_model(
    model_name: str,
    device: str,
    use_campp: bool,
    campp_similarity_threshold: float,
):
    from funasr import AutoModel

    key = (model_name, device, use_campp, campp_similarity_threshold)
    if key not in _MODEL_CACHE:
        model = AutoModel(
            model=model_name,
            vad_model="fsmn-vad",
            vad_kwargs={"max_single_segment_time": 30000},
            punc_model="ct-punc" if use_campp else None,
            spk_model="cam++" if use_campp else None,
            device="cuda:0" if device == "cuda" else "cpu",
            disable_update=True,
        )
        if use_campp:
            model.cb_model = _SmallSampleCamppCluster(campp_similarity_threshold)
        _MODEL_CACHE[key] = model
    return _MODEL_CACHE[key]


def _generate_with_speaker_turn_capture(model: Any, **kwargs: Any):
    """Capture FunASR's fine-grained diarization turns before sentence collapse."""

    if getattr(model, "spk_model", None) is None:
        return model.generate(**kwargs), []
    import funasr.auto.auto_model as auto_model_module

    original_distribute_spk = auto_model_module.distribute_spk
    captured_turns: list[list[Any]] = []

    def capture(sentence_list, speaker_turns):
        captured_turns[:] = [list(item) for item in speaker_turns]
        return original_distribute_spk(sentence_list, speaker_turns)

    auto_model_module.distribute_spk = capture
    try:
        return model.generate(**kwargs), captured_turns
    finally:
        auto_model_module.distribute_spk = original_distribute_spk


def _transcribe_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    audio_path = Path(str(payload["audio_path"]))
    model_name = str(payload.get("model_name") or "iic/SenseVoiceSmall")
    device = str(payload.get("device") or "cuda")
    language = str(payload.get("language") or "auto")
    use_campp = bool(payload.get("use_campp", True))
    campp_similarity_threshold = float(
        payload.get("campp_similarity_threshold") or 0.78
    )
    model = _load_model(
        model_name,
        device,
        use_campp,
        campp_similarity_threshold,
    )
    result, raw_speaker_turns = _generate_with_speaker_turn_capture(
        model,
        input=str(audio_path),
        cache={},
        language=language,
        use_itn=True,
        batch_size_s=60,
        # Keep VAD turns separate. Merging a 15-second clip would make CAM++
        # see several speakers as one embedding and destroy attribution.
        merge_vad=False,
        output_timestamp=True,
        return_raw_text=True,
    )
    raw = result[0] if result else {}
    if not isinstance(raw, Mapping):
        raw = {"text": str(raw)}
    sentence_info = raw.get("sentence_info") or []
    clusterer = model.cb_model if use_campp else None
    speaker_turns, speaker_label_map = normalize_speaker_turns(
        raw_speaker_turns
    )
    for item in sentence_info:
        if not isinstance(item, Mapping) or item.get("spk") is None:
            continue
        try:
            speaker_key: Any = int(item["spk"])
        except (TypeError, ValueError):
            speaker_key = str(item["spk"])
        if speaker_key not in speaker_label_map:
            speaker_label_map[speaker_key] = len(speaker_label_map)
    segments = sentence_info_to_segments(
        [item for item in sentence_info if isinstance(item, Mapping)],
        speaker_turns,
        speaker_label_map,
    )
    if not segments and raw.get("timestamp") and raw.get("words"):
        timestamps = raw["timestamp"]
        words = raw["words"]
        for pair, word in zip(timestamps, words):
            if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                continue
            segments.append(
                {
                    "start_sec": float(pair[0]) / 1000.0,
                    "end_sec": float(pair[1]) / 1000.0,
                    "text": str(word),
                    "speaker": None,
                }
            )
    return {
        "ok": True,
        "language": language,
        "raw_text": _flatten_text(raw.get("raw_text") or raw.get("text") or ""),
        "segments": segments,
        "speaker_turns": speaker_turns,
        "raw_sentence_info": _jsonable(sentence_info),
        "speaker_clustering": {
            "backend": "cam++-spectral-short-clip" if use_campp else "disabled",
            "similarity_threshold": campp_similarity_threshold,
            "embedding_count": len(getattr(clusterer, "last_labels", ())),
            "embedding_labels": list(getattr(clusterer, "last_labels", ())),
            "embedding_cluster_count": len(
                set(getattr(clusterer, "last_labels", ()))
            ),
            "speaker_turn_count": len(speaker_turns),
            "turn_speaker_count": len(
                {item["speaker"] for item in speaker_turns}
            ),
            "sentence_speaker_count": len(
                {
                    item.get("spk")
                    for item in sentence_info
                    if isinstance(item, Mapping) and item.get("spk") is not None
                }
            ),
            "granularity_conflict": bool(speaker_turns) and len(
                {item["speaker"] for item in speaker_turns}
            )
            > len(
                {
                    item.get("spk")
                    for item in sentence_info
                    if isinstance(item, Mapping) and item.get("spk") is not None
                }
            ),
            "raw_to_anonymous_label": {
                str(raw): label for raw, label in speaker_label_map.items()
            },
        },
    }


def _flatten_token_ids(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        flattened: list[int] = []
        for item in value:
            flattened.extend(_flatten_token_ids(item))
        return flattened
    return [int(value)]


def _encode_candidate(tokenizer: Any, text: str) -> list[int]:
    token_ids: list[int] = []
    for item in tokenizer.tokens2ids(tokenizer.text2tokens(text)):
        if isinstance(item, (list, tuple)) and not item:
            # SenseVoice uses 124 as its unknown-token fallback in the same
            # timestamp/forced-alignment path shipped by FunASR.
            token_ids.append(124)
        else:
            token_ids.extend(_flatten_token_ids(item))
    return token_ids


def _ctc_candidate_score(
    log_probs: Any,
    *,
    tokenizer: Any,
    text: str,
    blank_id: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    token_ids = _encode_candidate(tokenizer, text)
    if not token_ids:
        raise ValueError("candidate text has no SenseVoice tokens")
    target = torch.tensor(token_ids, dtype=torch.long, device=log_probs.device)
    loss = functional.ctc_loss(
        log_probs.unsqueeze(1),
        target,
        torch.tensor(
            [log_probs.size(0)],
            dtype=torch.long,
            device=log_probs.device,
        ),
        torch.tensor(
            [len(token_ids)],
            dtype=torch.long,
            device=log_probs.device,
        ),
        blank=blank_id,
        reduction="sum",
        zero_infinity=False,
    )
    log_likelihood = -float(loss.item())
    if not torch.isfinite(loss):
        raise ValueError("candidate CTC likelihood is not finite")
    return {
        "text": text,
        "token_count": len(token_ids),
        "ctc_log_likelihood": round(log_likelihood, 6),
        "normalized_log_likelihood": round(
            log_likelihood / max(1, len(token_ids)),
            6,
        ),
    }


def _pronunciation_signature(text: str) -> list[str]:
    """Return context-aware Mandarin syllables while preserving other text."""

    from pypinyin import Style, lazy_pinyin

    return [
        str(item).casefold()
        for item in lazy_pinyin(
            text,
            style=Style.TONE3,
            neutral_tone_with_five=True,
            errors=lambda value: list(value),
        )
    ]


def _score_candidate_clip(
    model: Any,
    audio: Any,
    sample_rate: int,
    candidate: Mapping[str, Any],
    *,
    language: str,
) -> dict[str, Any]:
    import numpy as np
    import torch

    candidate_id = str(candidate.get("candidate_id") or "")
    observed_text = str(candidate.get("observed_text") or "")
    expected_text = str(candidate.get("expected_text") or "")
    start_sec = float(candidate["start_sec"])
    end_sec = float(candidate["end_sec"])
    if not candidate_id:
        raise ValueError("candidate_id is required")
    if not observed_text or not expected_text:
        raise ValueError("observed_text and expected_text are required")
    if start_sec < 0 or end_sec <= start_sec:
        raise ValueError("candidate time range is invalid")

    duration_sec = len(audio) / sample_rate
    padding_sec = 0.08
    clip_start = max(0.0, start_sec - padding_sec)
    clip_end = min(duration_sec, end_sec + padding_sec)
    start_sample = int(round(clip_start * sample_rate))
    end_sample = int(round(clip_end * sample_rate))
    clip = np.asarray(audio[start_sample:end_sample], dtype="float32")
    if clip.size < max(1, int(sample_rate * 0.12)):
        raise ValueError("candidate audio clip is too short")

    ctc_projection = getattr(getattr(model.model, "ctc", None), "ctc_lo", None)
    if ctc_projection is None:
        raise RuntimeError("SenseVoice CTC projection is unavailable")
    captured_logits: list[Any] = []

    def capture_logits(_module: Any, _inputs: Any, output: Any) -> None:
        if not captured_logits:
            captured_logits.append(output.detach().float().cpu())

    hook = ctc_projection.register_forward_hook(capture_logits)
    try:
        inference_result = model.inference(
            clip,
            model=model.model,
            kwargs=model.kwargs,
            key=[candidate_id],
            language=language,
            use_itn=True,
            output_timestamp=False,
            data_type="sound",
            fs=sample_rate,
            batch_size=1,
            disable_pbar=True,
        )
    finally:
        hook.remove()
    if not captured_logits:
        raise RuntimeError("SenseVoice did not expose CTC logits")
    logits = captured_logits[0]
    if logits.ndim == 3:
        logits = logits[0]
    # SenseVoice prepends language, event, emotion, and text-normalization
    # query frames.  They are not part of either plain-text candidate.
    log_probs = torch.log_softmax(logits, dim=-1)[4:]
    if log_probs.numel() == 0:
        raise RuntimeError("SenseVoice CTC logits contain no speech frames")
    tokenizer = model.kwargs.get("tokenizer")
    if tokenizer is None:
        raise RuntimeError("SenseVoice tokenizer is unavailable")
    decoded_text = ""
    if inference_result and isinstance(inference_result[0], Mapping):
        decoded_text = str(inference_result[0].get("text") or "")
    blank_id = int(getattr(model.model, "blank_id", 0))
    observed_pronunciation = _pronunciation_signature(observed_text)
    expected_pronunciation = _pronunciation_signature(expected_text)
    return {
        "candidate_id": candidate_id,
        "clip_start_sec": round(clip_start, 6),
        "clip_end_sec": round(clip_end, 6),
        "sample_rate": sample_rate,
        "decoded_text": decoded_text,
        "pronunciation_relation": (
            "same_pronunciation"
            if observed_pronunciation == expected_pronunciation
            else "different_pronunciation"
        ),
        "observed_pronunciation": observed_pronunciation,
        "expected_pronunciation": expected_pronunciation,
        "observed": _ctc_candidate_score(
            log_probs,
            tokenizer=tokenizer,
            text=observed_text,
            blank_id=blank_id,
        ),
        "expected": _ctc_candidate_score(
            log_probs,
            tokenizer=tokenizer,
            text=expected_text,
            blank_id=blank_id,
        ),
    }


def _score_candidates_request(payload: Mapping[str, Any]) -> dict[str, Any]:
    import numpy as np
    import soundfile

    audio_path = Path(str(payload["audio_path"]))
    model_name = str(payload.get("model_name") or "iic/SenseVoiceSmall")
    device = str(payload.get("device") or "cuda")
    language = str(payload.get("language") or "auto")
    use_campp = bool(payload.get("use_campp", True))
    campp_similarity_threshold = float(
        payload.get("campp_similarity_threshold") or 0.78
    )
    candidates = payload.get("candidates") or ()
    if not isinstance(candidates, (list, tuple)):
        raise ValueError("candidates must be an array")
    if len(candidates) > 32:
        raise ValueError("too many constrained ASR candidates")
    audio, sample_rate = soundfile.read(
        str(audio_path),
        dtype="float32",
        always_2d=True,
    )
    audio = np.asarray(audio, dtype="float32").mean(axis=1)
    sample_rate = int(sample_rate)
    if sample_rate <= 0 or audio.size == 0:
        raise ValueError("candidate scoring audio is empty")
    model = _load_model(
        model_name,
        device,
        use_campp,
        campp_similarity_threshold,
    )
    scores = []
    for item in candidates:
        if not isinstance(item, Mapping):
            scores.append({"candidate_id": "", "error": "candidate is not an object"})
            continue
        candidate_id = str(item.get("candidate_id") or "")
        try:
            scores.append(
                _score_candidate_clip(
                    model,
                    audio,
                    sample_rate,
                    item,
                    language=language,
                )
            )
        except Exception as exc:
            scores.append(
                {
                    "candidate_id": candidate_id,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    return {
        "ok": True,
        "backend": "sensevoice_constrained_ctc",
        "model": model_name,
        "device": device,
        "language": language,
        "scores": scores,
    }


def _request(payload: Mapping[str, Any]) -> dict[str, Any]:
    action = str(payload.get("action") or "transcribe")
    if action == "transcribe":
        return _transcribe_request(payload)
    if action == "score_candidates":
        return _score_candidates_request(payload)
    raise ValueError(f"unsupported SenseVoice worker action: {action}")


def main() -> int:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError("请求必须是 JSON 对象")
            response = _request(payload)
        except Exception as exc:  # worker protocol must always return one line
            traceback.print_exc(file=sys.stderr)
            response = {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        print(
            json.dumps(response, ensure_ascii=False),
            file=_PROTOCOL_STDOUT,
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
