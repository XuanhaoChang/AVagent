"""Source-anchored speaking-role constraints mined from free-form prompts."""

from __future__ import annotations

import re
from typing import Any


_QUOTED_TEXT = re.compile(r"[\"“](?P<text>[^\"”\n]{1,240})[\"”]")
_AT_ROLE = re.compile(r"@(?P<role>[\w\u3400-\u9fff·]{1,24})")
_NAMED_SPEECH = re.compile(
    r"(?P<role>[\u3400-\u9fffA-Za-z][\w\u3400-\u9fff·]{0,15})"
    r"(?:\s*[（(][^）)]{0,30}[）)])?\s*"
    r"(?:说(?:道)?|问(?:道)?|回答|喊(?:道)?|低声说|大声说)?\s*[：:]\s*$"
)
_TIME_RANGE = re.compile(
    r"(?P<start>\d+(?:\.\d+)?)\s*(?:s|秒)?\s*[-~至]\s*"
    r"(?P<end>\d+(?:\.\d+)?)\s*(?:s|秒)"
)
_DIALOGUE_MARKERS = ("dialogue", "台词", "对白", "说：", "说道：", "说:")
_CLOSED_SCOPE_MARKERS = (
    "dialogue：无",
    "dialogue:无",
    "台词：无",
    "台词:无",
    "对白：无",
    "对白:无",
    "不得有其他台词",
    "不要有其他台词",
    "无其他台词",
)
_OPEN_SCOPE_MARKERS = ("自由发挥", "可自行发挥", "可即兴", "例如", "等台词")


def _nearest_time_range(prompt: str, quote_start: int) -> tuple[float, float] | None:
    section_start = max(0, quote_start - 900)
    matches = list(_TIME_RANGE.finditer(prompt[section_start:quote_start]))
    if not matches:
        return None
    match = matches[-1]
    start = float(match.group("start"))
    end = float(match.group("end"))
    return (start, end) if end > start else None


def _nearest_explicit_role(prompt: str, quote_start: int) -> tuple[str, str] | None:
    context_start = max(0, quote_start - 220)
    context = prompt[context_start:quote_start]
    lowered = context.casefold()
    has_dialogue_context = any(
        marker in lowered for marker in _DIALOGUE_MARKERS
    )
    at_mentions = list(_AT_ROLE.finditer(context))
    if at_mentions and has_dialogue_context:
        return at_mentions[-1].group("role"), "dialogue_nearest_at_role"
    named = _NAMED_SPEECH.search(context)
    if named is not None:
        role = named.group("role")
        for suffix in ("低声说", "大声说", "说道", "喊道", "回答", "说", "问"):
            if role.endswith(suffix) and len(role) > len(suffix):
                role = role[: -len(suffix)]
                break
        return role, "named_speech_before_quote"
    return None


def extract_prompt_speech_plan(user_prompt: str) -> dict[str, Any]:
    """Extract only explicit role-dialogue assignments with source spans.

    The extractor intentionally abstains on unlabelled quotations.  Its output
    is a target/evaluability plan, never acoustic ground truth and never a hard
    CAM++ cluster count.
    """

    prompt = str(user_prompt or "")
    turns: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for match in _QUOTED_TEXT.finditer(prompt):
        role_match = _nearest_explicit_role(prompt, match.start())
        if role_match is None:
            continue
        role, rule = role_match
        text = match.group("text").strip()
        key = (role, match.start("text"), match.end("text"))
        if not text or key in seen:
            continue
        seen.add(key)
        time_range = _nearest_time_range(prompt, match.start())
        turn: dict[str, Any] = {
            "role": role,
            "dialogue_text": text,
            "prompt_start": match.start("text"),
            "prompt_end": match.end("text"),
            "prompt_source_text": prompt[match.start("text") : match.end("text")],
            "extraction_rule": rule,
            "confidence": "high",
        }
        if time_range is not None:
            turn["expected_start_sec"] = time_range[0]
            turn["expected_end_sec"] = time_range[1]
        turns.append(turn)

    roles: list[str] = []
    for turn in turns:
        role = str(turn["role"])
        if role not in roles:
            roles.append(role)
    lowered_prompt = prompt.casefold().replace(" ", "")
    if not turns:
        scope = "none"
    elif any(marker in lowered_prompt for marker in _OPEN_SCOPE_MARKERS):
        scope = "partial"
    elif (
        any(marker in lowered_prompt for marker in _CLOSED_SCOPE_MARKERS)
        or ("镜号" in prompt and ("dialogue" in lowered_prompt or "台词" in prompt))
    ):
        scope = "closed"
    else:
        scope = "partial"
    return {
        "version": 1,
        "method": "source_anchored_explicit_role_dialogue_extraction",
        "scope": scope,
        "confidence": "high" if turns else "none",
        "expected_speaking_roles": roles,
        "expected_speaker_count": len(roles),
        "allows_unassigned_speech": scope != "closed",
        "turns": turns,
    }
