"""Text-only GPT review of input metadata, gold annotations, and predictions."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


PREDICTION_SOURCES = ("gpt_a", "gpt_b", "seed_a", "seed_b", "seed_c")
CATEGORY_NAMES = {
    1: "完整指出了GT的问题",
    2: "指出了GT的问题且指出了别的问题",
    3: "没有完全指出GT的问题",
    4: "GT问题超出评测框架",
    5: "输入材料不足",
}
COVERAGE_STATUSES = {
    "covered",
    "partial",
    "missed",
    "not_applicable",
    "insufficient_evidence",
}
CONFIDENCE_VALUES = {"高", "中", "低"}


SYSTEM_PROMPT = """\
你是结构化评测结果的文本复核员。你会收到 input JSON、材料审计、GT JSON 和预测 JSON。
你没有实际查看或收听任何媒体，只能用 input JSON 判断 prompt 依赖了哪些输入材料、
对应字段是否提供；不得把媒体路径当作已经查看过的内容，也不得用 user_prompt 或用户反馈
替代 GT、证明预测正确或补充视频事实。

请分别对五个 prediction_source 分类：
1. 完整指出了 GT 中所有可回答的问题，且没有提出 GT 之外的独立问题。
2. 完整指出了 GT 中所有可回答的问题，同时还提出了 GT 之外的独立问题。
3. 至少一个本可由该评测框架回答的 GT 问题被遗漏、只部分覆盖或核心事实不一致。
4. GT 描述的不是可由“结合明确指令、参考约束和生成音视频证据”回答的问题，
   例如模型降智、抽卡概率、提示词优化、产品功能、审核或纯主观诉求。
5. GT 问题原则上属于框架范围，但缺失的输入材料直接导致该 GT 问题无法判断。
   例如 GT 要求核对参考音色而参考音频缺失，或 GT 要求比较参考视频而参考视频缺失。

判定规则：
- 材料审计中的 missing_required_materials 非空只表示发现了 prompt 依赖素材缺口，不能
  自动把任何 prediction_source 归为类别 5。必须逐个 GT 问题判断该缺口是否使它无法判断。
- 如果缺失素材与某个 GT 问题无关，仍按可回答问题正常判断预测覆盖：类别 1、2 或 3。
- 如果 GT 同时包含可判断和因材料缺失而无法判断的问题，不得把所有问题或所有来源一律改成
  类别 5；对可判断问题继续按覆盖情况分类，并在对应 gt_coverage 中标记材料不足的问题。
- 只有当材料缺失确实阻断了 GT 所指出问题的判断时，才使用类别 5；材料缺口本身不是类别 5
  的充分条件。
- prompt 提到“参考视频”“视频1”等，但 input 没有非空参考视频字段时，属于缺少参考视频；
  generated_video_url 是待评估生成视频，不能当作参考视频。
- prompt 提到“参考音频”“音频1”“某角色的音色”等外部声音素材，但 input 没有非空
  参考音频字段时，属于缺少参考音频；reference_image_urls 只能证明提供了参考图片。
- 不能因为 prompt 声称“已提供”就认定素材实际存在，必须以 input 的非空字段为准。
- 类别 4、5 描述 GT 本身的可回答性；若一条样本的全部 GT 均属于同一种情况，
  五个预测来源通常应得到相同的 4 或 5。
- 若 GT 同时包含可回答与不可回答问题，仍检查预测对可回答 GT 的覆盖，通常归 1、2 或 3。
- “别的问题”只按预测中独立于 GT 的问题点计数；不得判断它在真实视频中是否正确。
- 语义一致即可，不要求问题类型标签、时间或 BBox 文本完全相同。
- 不得把预测声称“无法验证”自动视为覆盖 GT。
- 每个 GT 问题都必须给出 coverage 记录。
- 比较多个来源时，如果一个预测完整包含另一个预测的全部问题且只追加新问题，
  其 GT 覆盖不能更差；追加 GT 外问题只能使类别 1 变为类别 2，
  追加内容覆盖原遗漏时则可以使类别 3 改善为类别 1 或 2。

只输出一个 JSON 对象，不输出 Markdown。字段必须为：
{
  "sample_id": "...",
  "reviews": [
    {
      "prediction_source": "gpt_a",
      "category": 1,
      "reason": "一句简短、可审计的分类依据",
      "gt_coverage": [
        {
          "gt_index": 1,
          "status": "covered|partial|missed|not_applicable|insufficient_evidence",
          "matched_prediction_indices": [1],
          "reason": "简短语义对照"
        }
      ],
      "extra_prediction_indices": [],
      "confidence": "高|中|低"
    }
  ]
}
reviews 必须按 gpt_a、gpt_b、seed_a、seed_b、seed_c 顺序各出现一次。
"""


def _prediction_sources(
    prediction_sources: tuple[str, ...],
) -> tuple[str, ...]:
    if not prediction_sources or len(set(prediction_sources)) != len(
        prediction_sources
    ):
        raise ValueError("预测来源必须非空且不能重复")
    return prediction_sources


def system_prompt_for_sources(prediction_sources: tuple[str, ...]) -> str:
    sources = _prediction_sources(prediction_sources)
    source_list = "、".join(sources)
    prompt = SYSTEM_PROMPT.replace(
        '"prediction_source": "gpt_a"',
        f'"prediction_source": "{sources[0]}"',
    )
    prompt = prompt.replace(
        "请分别对五个 prediction_source 分类：",
        f"请分别对以下 prediction_source 分类：{source_list}。",
    )
    prompt = prompt.replace(
        "五个预测来源通常应得到相同的 4 或 5。",
        "所有预测来源通常应得到相同的 4 或 5。",
    )
    return prompt.replace(
        "reviews 必须按 gpt_a、gpt_b、seed_a、seed_b、seed_c 顺序各出现一次。",
        f"reviews 必须按 {source_list} 顺序各出现一次。",
    )


def _object_array(value: Any, source: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{source} 必须是 JSON 对象数组")
    return value


REFERENCE_VIDEO_PATTERN = re.compile(
    r"(?:参考\s*(?:视频|影片)|"
    r"[@<【]\s*(?:视频|影片)\s*[0-9一二三四五六七八九十]+|"
    r"(?:使用|以|按照|根据|依照|仿照|模仿)\s*"
    r"(?:视频|影片)\s*[0-9一二三四五六七八九十]+|"
    r"(?:编辑|修改|替换|保留)\s*[@<【]?\s*(?:视频|影片)\s*"
    r"[0-9一二三四五六七八九十]+|"
    r"(?:视频|影片)\s*[0-9一二三四五六七八九十]+\s*(?:是|为)"
    r".{0,40}(?:参考|素材|产品|原始|固定))",
    re.IGNORECASE,
)
REFERENCE_AUDIO_PATTERN = re.compile(
    r"(?:参考\s*(?:音频|声音|音色)|音色\s*参考|"
    r"[@<【]\s*(?:音频|声音素材)\s*[0-9一二三四五六七八九十]+|"
    r"(?:使用|以|按照|根据|依照|仿照|模仿)\s*"
    r"(?:音频|声音素材)\s*[0-9一二三四五六七八九十]+|"
    r"(?:音频|声音素材)\s*[0-9一二三四五六七八九十]+\s*(?:是|为)"
    r".{0,40}(?:参考|素材|音色|声音)|"
    r"(?:音色|声音)\s*(?:按照|使用|参考|采用|来自).{0,20}"
    r"(?:音频|声音素材)\s*[0-9一二三四五六七八九十]+|"
    r"图\s*[0-9一二三四五六七八九十]+音频\s*"
    r"[0-9一二三四五六七八九十]+)",
    re.IGNORECASE,
)


def _has_nonempty_value(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple, set)):
        return any(_has_nonempty_value(item) for item in value)
    if isinstance(value, dict):
        return any(_has_nonempty_value(item) for item in value.values())
    return True


def _has_reference_material(input_data: dict[str, Any], kind: str) -> bool:
    aliases = {
        "video": {
            "reference_video_url",
            "reference_video_urls",
            "reference_videos",
            "参考视频",
            "参考视频地址",
        },
        "audio": {
            "reference_audio_url",
            "reference_audio_urls",
            "reference_audios",
            "audio_urls",
            "input_audio_urls",
            "参考音频",
            "参考音频地址",
        },
    }[kind]
    tokens = {
        "video": ("video", "视频", "影片"),
        "audio": ("audio", "音频", "声音", "音色"),
    }[kind]
    for key, value in input_data.items():
        normalized = str(key).strip().lower()
        is_explicit_alias = normalized in aliases
        is_reference_key = (
            ("reference" in normalized or "参考" in normalized)
            and any(token in normalized for token in tokens)
        )
        if (is_explicit_alias or is_reference_key) and _has_nonempty_value(value):
            return True
    return False


def missing_required_materials(input_data: dict[str, Any]) -> tuple[str, ...]:
    if not isinstance(input_data, dict):
        raise ValueError("input 必须是 JSON 对象")
    prompt = str(input_data.get("user_prompt", ""))
    missing: list[str] = []
    if REFERENCE_VIDEO_PATTERN.search(prompt) and not _has_reference_material(
        input_data, "video"
    ):
        missing.append("参考视频")
    if REFERENCE_AUDIO_PATTERN.search(prompt) and not _has_reference_material(
        input_data, "audio"
    ):
        missing.append("参考音频")
    return tuple(missing)


def review_input_context(input_data: dict[str, Any]) -> dict[str, Any]:
    images = input_data.get("reference_image_urls")
    if isinstance(images, list):
        image_count = sum(_has_nonempty_value(item) for item in images)
    else:
        image_count = int(_has_nonempty_value(images))
    return {
        "序号": input_data.get("序号", ""),
        "user_prompt": str(input_data.get("user_prompt", "")),
        "provided_materials": {
            "reference_image_count": image_count,
            "reference_video_provided": _has_reference_material(
                input_data, "video"
            ),
            "reference_audio_provided": _has_reference_material(
                input_data, "audio"
            ),
            "generated_video_provided": _has_nonempty_value(
                input_data.get("generated_video_url")
            ),
        },
    }


def read_sample(
    sample_dir: Path,
    prediction_sources: tuple[str, ...] = PREDICTION_SOURCES,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    dict[str, list[dict[str, Any]]],
]:
    sources = _prediction_sources(prediction_sources)
    input_data = json.loads(
        (sample_dir / "input.json").read_text(encoding="utf-8")
    )
    if not isinstance(input_data, dict):
        raise ValueError(f"{sample_dir.name}/input.json 必须是 JSON 对象")
    gt = _object_array(
        json.loads((sample_dir / "gt.json").read_text(encoding="utf-8")),
        f"{sample_dir.name}/gt.json",
    )
    predictions = {
        source: _object_array(
            json.loads((sample_dir / f"{source}.json").read_text(encoding="utf-8")),
            f"{sample_dir.name}/{source}.json",
        )
        for source in sources
    }
    return input_data, gt, predictions


def build_messages(
    sample_id: str,
    input_data: dict[str, Any],
    gt: list[dict[str, Any]],
    predictions: dict[str, list[dict[str, Any]]],
    prediction_sources: tuple[str, ...] = PREDICTION_SOURCES,
) -> list[dict[str, str]]:
    sources = _prediction_sources(prediction_sources)
    if tuple(predictions) != sources:
        raise ValueError("预测来源必须按固定顺序提供")
    if not isinstance(input_data, dict):
        raise ValueError("input 必须是 JSON 对象")
    missing_materials = missing_required_materials(input_data)
    payload = {
        "sample_id": sample_id,
        "input": review_input_context(input_data),
        "material_audit": {
            "missing_required_materials": list(missing_materials),
            "force_category": None,
            "missing_materials_are_not_auto_category_5": True,
        },
        "gt": _object_array(gt, "gt"),
        "predictions": {
            source: _object_array(predictions[source], source)
            for source in sources
        },
    }
    return [
        {"role": "system", "content": system_prompt_for_sources(sources)},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        stripped = stripped.rsplit("```", 1)[0].strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("API 未返回 JSON 对象")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("API 返回值顶层必须是 JSON 对象")
    return value


def parse_review_response(
    text: str,
    expected_sample_id: str,
    prediction_sources: tuple[str, ...] = PREDICTION_SOURCES,
) -> dict[str, Any]:
    expected_sources = _prediction_sources(prediction_sources)
    value = _extract_json_object(text)
    if value.get("sample_id") != expected_sample_id:
        raise ValueError("API 返回的 sample_id 不匹配")
    reviews = value.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("reviews 必须是数组")
    sources = [item.get("prediction_source") for item in reviews if isinstance(item, dict)]
    if sources != list(expected_sources):
        raise ValueError("reviews 的预测来源缺失、重复或顺序错误")

    normalized_reviews = []
    for item in reviews:
        category = item.get("category")
        if category not in CATEGORY_NAMES:
            raise ValueError(f"{item['prediction_source']} 的 category 必须为 1-5")
        reason = str(item.get("reason", "")).strip()
        if not reason:
            raise ValueError(f"{item['prediction_source']} 缺少 reason")
        coverage = item.get("gt_coverage")
        if not isinstance(coverage, list) or not all(isinstance(row, dict) for row in coverage):
            raise ValueError(f"{item['prediction_source']} 的 gt_coverage 无效")
        for row in coverage:
            if row.get("status") not in COVERAGE_STATUSES:
                raise ValueError(f"{item['prediction_source']} 的 coverage status 无效")
            if not isinstance(row.get("matched_prediction_indices", []), list):
                raise ValueError(f"{item['prediction_source']} 的匹配索引无效")
        extra = item.get("extra_prediction_indices")
        if not isinstance(extra, list) or not all(isinstance(index, int) for index in extra):
            raise ValueError(f"{item['prediction_source']} 的额外问题索引无效")
        confidence = item.get("confidence")
        if confidence not in CONFIDENCE_VALUES:
            raise ValueError(f"{item['prediction_source']} 的 confidence 无效")
        normalized_reviews.append(
            {
                "prediction_source": item["prediction_source"],
                "category": category,
                "category_name": CATEGORY_NAMES[category],
                "reason": reason,
                "gt_coverage": coverage,
                "extra_prediction_indices": extra,
                "confidence": confidence,
            }
        )
    return {"sample_id": expected_sample_id, "reviews": normalized_reviews}


def response_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(
            str(part.get("text", ""))
            for part in content
            if isinstance(part, dict) and part.get("type") in {"text", "output_text"}
        ).strip()
    return ""


def chat_completion(
    *,
    api_url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
    max_attempts: int,
) -> tuple[str, dict[str, Any], int]:
    payload = {
        "model": model,
        "messages": messages,
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    last_error: BaseException | None = None
    for attempt in range(1, max(1, max_attempts) + 1):
        request = urllib.request.Request(
            api_url,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
            message = result["choices"][0]["message"]
            return response_text(message.get("content")), result.get("usage", {}), len(body)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            last_error = RuntimeError(f"Chat Completions HTTP {exc.code}: {detail}")
            if exc.code != 429 and not 500 <= exc.code < 600:
                raise last_error from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError, IndexError) as exc:
            last_error = exc
        if attempt < max(1, max_attempts):
            time.sleep(min(8, 2 ** (attempt - 1)))
    raise RuntimeError(f"Chat Completions 请求失败：{last_error}") from last_error
