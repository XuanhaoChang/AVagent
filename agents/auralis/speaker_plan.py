"""Source-anchored speaking-role constraints mined from free-form prompts."""

from __future__ import annotations

import re
from typing import Any, Sequence


_QUOTED_TEXT = re.compile(r"[\"“](?P<text>[^\"”\n]{1,240})[\"”]")
_UNQUOTED_SPEECH_TEXT = re.compile(
    r"(?:说(?:道|到)?|问(?:道)?|回答|喊(?:道)?|低声说|大声说|急喊|高喊|大喊)"
    r"\s*[：:]\s*(?P<text>[^\n]{1,240})"
)
_AT_ROLE = re.compile(r"@(?P<role>[\w\u3400-\u9fff·]{1,24})")
_AT_IMAGE_ROLE_ASSIGNMENT = re.compile(
    r"@(?:图片|图)\s*(?P<reference_index>\d+)\s*"
    r"(?:为|是|=)\s*(?P<role>[\w\u3400-\u9fff·]{1,24})"
)
_INLINE_IMAGE_ROLE = re.compile(
    r"(?P<role>[\u3400-\u9fffA-Za-z·]{2,12})\s*"
    r"(?:【\s*)?(?:图片|图)\s*(?P<reference_index>\d+)\s*】?"
)
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
_ROLE_TABLE = re.compile(
    r"视频角色对照表\s*[:：](?P<body>.*?)"
    r"(?=\n\s*视频(?:场景|角色|道具)对照表|\Z)",
    re.DOTALL,
)
_ROLE_ASSIGNMENT = re.compile(
    r"(?:^|[；;，,])\s*(?P<role>[\w\u3400-\u9fff·]{1,24})\s*="
)
_ROLE_REFERENCE_ASSIGNMENT = re.compile(
    r"(?:^|[；;，,])\s*(?P<role>[\w\u3400-\u9fff·]{1,24})\s*=\s*"
    r"【?\s*图\s*(?P<reference_index>\d+)\s*】?"
)
_LOGIC_ROLE = re.compile(
    r"(?:【|[；;])\s*(?P<role>[\w\u3400-\u9fff·]{1,24}?)处于"
)
_SPEECH_CUE = re.compile(
    r"(?P<cue>厉声喝道|大声喝道|高声喝斥|颤着声问|厉声大喝|"
    r"厉声道|低声道|说道|说到|问道|急喊|高喊|喊道|大喊|说|问|喊)"
    r"\s*[：:]\s*$"
)
_GENERIC_SPEECH_LABELS = {
    "厉声道",
    "低声道",
    "急喊",
    "高喊",
    "喊道",
    "大喊",
    "画外传来薛琴急促高喊",
}
_NON_CHARACTER_IMAGE_MARKERS = (
    "产品",
    "场景",
    "道具",
    "物品",
    "参考",
    "背景",
    "镜头",
    "画面",
    "位置",
)


def _nearest_time_range(prompt: str, quote_start: int) -> tuple[float, float] | None:
    section_start = max(0, quote_start - 900)
    matches = list(_TIME_RANGE.finditer(prompt[section_start:quote_start]))
    if not matches:
        return None
    match = matches[-1]
    start = float(match.group("start"))
    end = float(match.group("end"))
    return (start, end) if end > start else None


def _valid_inline_character_role(role: str) -> bool:
    return bool(role) and not any(
        marker in role for marker in _NON_CHARACTER_IMAGE_MARKERS
    )


def _canonical_role_phrase(
    role: str,
    known_roles: Sequence[str] = (),
) -> str:
    cleaned = re.sub(r"^(?:跟|和|与|及|由|为|是)", "", role.strip())
    cleaned = re.sub(r"(?:始终|一直|仍然|依旧|正在)$", "", cleaned)
    matches = [item for item in known_roles if item and item in cleaned]
    return max(matches, key=len) if matches else cleaned


def _known_prompt_role_aliases(prompt: str) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for match in _AT_IMAGE_ROLE_ASSIGNMENT.finditer(prompt):
        role = match.group("role")
        reference_index = int(match.group("reference_index"))
        aliases.setdefault(role, role)
        aliases.setdefault(f"图片{reference_index}", role)
        aliases.setdefault(f"图{reference_index}", role)
    for match in _AT_ROLE.finditer(prompt):
        role = match.group("role")
        if re.fullmatch(r"(?:图片|图)\d+", role):
            continue
        context = prompt[max(0, match.start() - 80) : match.end() + 40].casefold()
        if any(marker in context for marker in _DIALOGUE_MARKERS):
            aliases.setdefault(role, role)
    for table in _ROLE_TABLE.finditer(prompt):
        for match in _ROLE_ASSIGNMENT.finditer(table.group("body")):
            role = match.group("role")
            aliases.setdefault(role, role)
        for match in _ROLE_REFERENCE_ASSIGNMENT.finditer(table.group("body")):
            role = match.group("role")
            reference_index = int(match.group("reference_index"))
            aliases.setdefault(f"图片{reference_index}", role)
            aliases.setdefault(f"图{reference_index}", role)
    for match in _LOGIC_ROLE.finditer(prompt):
        role = _canonical_role_phrase(
            match.group("role"),
            tuple(dict.fromkeys(aliases.values())),
        )
        aliases.setdefault(role, role)
    for match in _INLINE_IMAGE_ROLE.finditer(prompt):
        role = _canonical_role_phrase(
            match.group("role"),
            tuple(dict.fromkeys(aliases.values())),
        )
        if not _valid_inline_character_role(role):
            continue
        reference_index = int(match.group("reference_index"))
        aliases.setdefault(role, role)
        aliases.setdefault(f"图片{reference_index}", role)
        aliases.setdefault(f"图{reference_index}", role)
    return aliases


def _known_prompt_roles(prompt: str) -> list[str]:
    roles: list[str] = []
    for role in _known_prompt_role_aliases(prompt).values():
        if role not in roles:
            roles.append(role)
    return roles


def _role_reference_images(prompt: str) -> dict[str, list[int]]:
    """Map roles to explicitly assigned reference-image indices.

    Only the role table is considered.  Scene and prop tables can also contain
    ``图N`` tokens, but they must never be used to infer character identity.
    """

    role_references: dict[str, list[int]] = {}
    for match in _AT_IMAGE_ROLE_ASSIGNMENT.finditer(prompt):
        role = match.group("role")
        reference_index = int(match.group("reference_index"))
        references = role_references.setdefault(role, [])
        if reference_index not in references:
            references.append(reference_index)
    for table in _ROLE_TABLE.finditer(prompt):
        for match in _ROLE_REFERENCE_ASSIGNMENT.finditer(table.group("body")):
            role = match.group("role")
            reference_index = int(match.group("reference_index"))
            references = role_references.setdefault(role, [])
            if reference_index not in references:
                references.append(reference_index)
    for match in _INLINE_IMAGE_ROLE.finditer(prompt):
        role = _canonical_role_phrase(
            match.group("role"),
            tuple(role_references),
        )
        if not _valid_inline_character_role(role):
            continue
        reference_index = int(match.group("reference_index"))
        references = role_references.setdefault(role, [])
        if reference_index not in references:
            references.append(reference_index)
    return role_references


def _action_speaker_role(prompt: str, quote_start: int) -> str | None:
    context_start = max(0, quote_start - 260)
    context = prompt[context_start:quote_start]
    cue_match = _SPEECH_CUE.search(context)
    if cue_match is None:
        return None
    role_aliases = _known_prompt_role_aliases(prompt)
    occurrences = [
        (context.rfind(alias), role)
        for alias, role in role_aliases.items()
        if context.rfind(alias) >= 0
    ]
    if not occurrences:
        return None
    cue = cue_match.group("cue")
    action_start = max(context.rfind("主体动作:"), context.rfind("主体动作："))
    if action_start >= 0:
        action_occurrences = [item for item in occurrences if item[0] > action_start]
        if action_occurrences:
            if cue in {"急喊", "高喊"}:
                return max(action_occurrences)[1]
            return min(action_occurrences)[1]
    clause_start = max(
        context.rfind(marker, 0, cue_match.start())
        for marker in ("。", "！", "？", "；", ";", "，", ",", "\n")
    )
    clause_occurrences = [item for item in occurrences if item[0] > clause_start]
    if clause_occurrences:
        # In constructions such as ``女主看着男主说`` or ``甲对乙说``, the
        # first role in the local clause is the speaking subject; the nearest
        # role is commonly the object being addressed.
        return min(clause_occurrences)[1]
    return max(occurrences)[1]


def _nearest_explicit_role(prompt: str, quote_start: int) -> tuple[str, str] | None:
    context_start = max(0, quote_start - 220)
    context = prompt[context_start:quote_start]
    lowered = context.casefold()
    has_dialogue_context = any(
        marker in lowered for marker in _DIALOGUE_MARKERS
    )
    at_mentions = list(_AT_ROLE.finditer(context))
    at_mentions = [
        match
        for match in at_mentions
        if not re.fullmatch(r"(?:图片|图)\d+", match.group("role"))
    ]
    if at_mentions and has_dialogue_context:
        return at_mentions[-1].group("role"), "dialogue_nearest_at_role"
    action_role = _action_speaker_role(prompt, quote_start)
    if action_role is not None:
        return action_role, "action_subject_before_speech_cue"
    named = _NAMED_SPEECH.search(context)
    if named is not None:
        role = named.group("role")
        for suffix in ("低声说", "大声说", "说道", "喊道", "回答", "说", "问"):
            if role.endswith(suffix) and len(role) > len(suffix):
                role = role[: -len(suffix)]
                break
        if role in _GENERIC_SPEECH_LABELS:
            return None
        role_aliases = _known_prompt_role_aliases(prompt)
        canonical_matches = [
            (role.find(alias), -len(alias), canonical)
            for alias, canonical in role_aliases.items()
            if role.find(alias) >= 0
        ]
        if canonical_matches:
            return min(canonical_matches)[2], "known_role_in_speech_subject"
        if role_aliases:
            # An explicit role table or image-role declaration is available;
            # do not invent a new role from an action/manner phrase.
            return None
        return role, "named_speech_before_quote"
    return None


def extract_prompt_speech_plan(user_prompt: str) -> dict[str, Any]:
    """Extract only explicit role-dialogue assignments with source spans.

    The extractor intentionally abstains on unlabelled quotations.  Its output
    is a target/evaluability plan, never acoustic ground truth and never a hard
    CAM++ cluster count.
    """

    prompt = str(user_prompt or "")
    role_reference_images = _role_reference_images(prompt)
    turns: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    quoted_matches = list(_QUOTED_TEXT.finditer(prompt))
    quoted_ranges = [(match.start(), match.end()) for match in quoted_matches]
    dialogue_matches = list(quoted_matches)
    for match in _UNQUOTED_SPEECH_TEXT.finditer(prompt):
        if any(start <= match.start("text") < end for start, end in quoted_ranges):
            continue
        if match.group("text").lstrip().startswith(("\"", "“")):
            continue
        dialogue_matches.append(match)
    dialogue_matches.sort(key=lambda item: item.start("text"))

    for match in dialogue_matches:
        dialogue_start = match.start("text")
        role_anchor = (
            match.start()
            if match.re is _QUOTED_TEXT
            else dialogue_start
        )
        role_match = _nearest_explicit_role(prompt, role_anchor)
        if role_match is None:
            continue
        role, rule = role_match
        if match.re is _UNQUOTED_SPEECH_TEXT and role not in _known_prompt_roles(prompt):
            # Unquoted free-form prose is too easy to parse as an action phrase
            # (for example "突然激动跑进来大喊") rather than a role name.
            continue
        text = match.group("text").strip().strip("\"“”")
        key = (role, match.start("text"), match.end("text"))
        if not text or key in seen:
            continue
        seen.add(key)
        time_range = _nearest_time_range(prompt, dialogue_start)
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
        reference_indices = role_reference_images.get(role, [])
        if reference_indices:
            turn["reference_image_indices"] = list(reference_indices)
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
        "version": 2,
        "method": "source_anchored_explicit_role_dialogue_extraction",
        "scope": scope,
        "confidence": "high" if turns else "none",
        "expected_speaking_roles": roles,
        "expected_speaker_count": len(roles),
        "allows_unassigned_speech": scope != "closed",
        "role_reference_images": role_reference_images,
        "turns": turns,
    }
