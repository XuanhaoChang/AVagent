# AVagent Agent Instructions

## Project Focus

This workspace centers on two related flows:

- `call_ffmpeg_skill.py`: runs row-by-row evaluation for `input/gt.csv` and writes `output/pred.csv`.
- `docs/build_week1_ppt.js`: generates the Week 1 progress deck and should stay aligned with the design/plan docs.

## Read First

- [Implementation plan](docs/superpowers/plans/2026-07-21-week1-progress-ppt.md)
- [PPT design spec](docs/superpowers/specs/2026-07-21-week1-progress-ppt-design.md)
- [Evaluation skill](SKILL.md)

Prefer linking to these docs instead of restating their contents.

## Working Rules

- Keep changes minimal and scoped to the requested task.
- Do not edit generated artifacts under `artifacts/` unless the task is explicitly about regeneration or verification.
- Preserve the CSV schema and row order in `call_ffmpeg_skill.py`; it must not expose `思考过程及标准答案` to the model.
- Treat the evaluation prompt as evidence-grounded: user feedback is a high-priority clue, not proof.
- Keep responses and user-facing content in Chinese unless the user asks otherwise.

## Validation

- For the deck generator, run `node docs/build_week1_ppt.js`, then verify the produced PPTX/PDF with the rendering steps described in the plan.
- For the evaluation script, prefer a narrow syntax/runtime check on `call_ffmpeg_skill.py` after edits.
- If a change touches prompt structure or output schema, re-check the corresponding docs before merging edits.

## Common Pitfalls

- `call_ffmpeg_skill.py` depends on local media paths, `ffmpeg`, and `ffprobe`; do not assume remote URLs are usable as-is.
- The deck content distinguishes completed research work from unverified experiments; do not blur that boundary.
- If repeated friction appears in this workspace, capture it in `/chronicle` or update these instructions so future agents do not rediscover it.
conda activate avagent
