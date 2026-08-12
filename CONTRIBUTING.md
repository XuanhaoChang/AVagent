# Contributing

Thank you for improving AVAgent.

## Setup

```bash
git clone --recurse-submodules https://github.com/XuanhaoChang/AVagent.git
cd AVagent
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -p 'test_*.py'
```

Runtime model stacks are optional for deterministic development. Install them
only when changing the corresponding integration.

## Pull requests

- Keep changes focused and explain their evidence or behavior impact.
- Add deterministic tests for behavior changes.
- Preserve the separation between tool status and defect decisions.
- Do not commit datasets, generated media, credentials, checkpoints, or local
  experiment artifacts.
- Do not patch the AVBench submodule in place; put AVAgent-specific adapters in
  this repository.
- Include the commands and results used for validation.

Use short, imperative commit subjects such as `Rename the public evaluator`.
