# Repository Guidelines

## Scope and safety

- Preserve unrelated uncommitted changes and inspect `git status` before edits.
- Never commit credentials, private datasets, model weights, generated media,
  review packages, or experiment outputs.
- Treat failed or unavailable evidence as `not_evaluable`, not as proof that a
  defect is absent.

## Project structure

- `run_avagent.py` is the public evaluation entry point.
- `agents/` contains specialist orchestration and evidence gates.
- `agents/classic_checks/` contains typed contracts and issue-oriented checks.
- `av_eval/` contains dataset, audit, routing, review, and CLI modules.
- `tools/` contains reusable local media and evidence tools.
- `third_party/AVBench` is an upstream submodule; project-specific adapters
  belong in this repository, not as uncommitted submodule patches.

## Development

Follow the surrounding Python style: four-space indentation, type hints where
practical, `pathlib.Path`, small helpers, and deterministic outputs. Add or
update a `unittest` test for behavior changes.

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -p 'test_*.py'
```

Keep local work in ignored `input/`, `output/`, `models/`, `.external/`, or
`.local/` directories. Before publishing, inspect staged paths explicitly and
run a credential scan.
