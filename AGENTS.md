# Repository Guidelines

## Project Structure

- `av_eval/` contains the core dataset, taxonomy, audit, routing, experiment, and CLI modules.
- `agents/auralis/` implements the local ASR/OCR plus Gemini audio-evaluation agent; shared media, transcription, and subtitle tools are under `tools/`.
- `scripts/` holds focused experiment, export, smoke-test, and upload entry points. Tests live in `tests/` and use `test_*.py` naming.
- `docs/evaluation/` contains execution and design notes. `input/`, `output/`, and `models/` are local data/artifact areas; `third_party/AVBench` is an external submodule. Large server-transfer assets are in `server_payload/` and tracked with Git LFS.

## Agent Architecture and Review Workflow

- `call_ffmpeg_skill_gpt_d.py` is the current Agent-D framework, developed incrementally for GPT-A plus Auralis audio/subtitle evidence and Gemini review; it now makes a final GPT text-only synthesis call to organize and deduplicate both results. It is the normal entry point for new evaluation runs.
- `call_ffmpeg_skill.py` is the original baseline implementation kept as a backup and comparison reference; do not treat it as the current architecture.
- `agents/auralis/` contains the sub-agent orchestration, local ASR/OCR/alignment, and Gemini gateway. `agents/avbench_sync.py` keeps a process-isolated AVBench worker alive; precise audio-video synchronization belongs to AVBench and does not require audio chunks in Agent-D.
- The installed AVBench runtime is `.conda-envs/avbench`; LatentSync is in `.external/LatentSync`, with the legacy `checkpoints/auxiliary/syncnet_v2.model` checkpoint. The normal ASR/OCR/Gemini runtime remains the `avagent` environment.
- The review tools are `scripts/review_text_predictions.py` (text reviewer) and `scripts/classify_samples_by_gpt_a.py` (five-category exporter). Sample packages are created by `scripts/export_human_review_samples.py`; additional predictions can be attached with `scripts/attach_prediction_to_review_samples.py`.

Use this order for an auditable run:

1. Run tests, then run Agent-D and save its CSV plus JSONL log under a task-specific directory, for example `output/captionErr/agentd/`.
2. Export review packages under `output/<task>/human_review_samples/`; attach the Agent-D CSV with `--label agentd` and its `--run-log` so each package also contains `agentd.json`, `asr.json`, and `ocr.json`.
3. Run the reviewer and write `results.jsonl` plus `summary.json` under `output/<task>/review/`.
4. Run the classifier with `--prediction-source agentd`; it copies complete sample packages into `output/<task>/human_review_samples_by_agentd/` and adds `agentd_review.json`.

Example review commands:

```bash
python call_ffmpeg_skill_gpt_d.py \
  --input-csv output/<task>/agentd_input.csv \
  --latentsync-root .external/LatentSync \
  --avbench-syncnet-ckpt .external/LatentSync/checkpoints/auxiliary/syncnet_v2.model \
  --avbench-python .conda-envs/avbench/bin/python \
  --output-csv output/<task>/agentd/pred.csv \
  --run-log output/<task>/agentd/run.jsonl
python scripts/attach_prediction_to_review_samples.py \
  --prediction-csv output/<task>/agentd/pred.csv \
  --samples-root output/<task>/human_review_samples \
  --label agentd \
  --run-log output/<task>/agentd/run.jsonl
python scripts/review_text_predictions.py \
  --input-root output/<task>/human_review_samples \
  --output-jsonl output/<task>/review/results.jsonl \
  --summary-json output/<task>/review/summary.json \
  --prediction-source agentd
python scripts/classify_samples_by_gpt_a.py \
  --samples-root output/<task>/human_review_samples \
  --reviews-jsonl output/<task>/review/results.jsonl \
  --prediction-source agentd \
  --output-root output/<task>/human_review_samples_by_agentd
```

## Build, Test, and Development Commands

Use the project environment before running Python commands:

```bash
conda activate avagent
python -m pip install -r requirements-audio-agent.txt
python -m unittest discover -s tests -p 'test_*.py'
python -m av_eval.cli capacity-plan --output /tmp/avagent-capacity.json
python scripts/smoke_auralis_local.py /path/to/video.mp4
```

The activation command selects the project environment; the install command adds Auralis dependencies; the test command runs the full suite; the CLI example generates a capacity plan; and the smoke test checks local media evidence without a remote model call. Use `conda run -n avagent ffmpeg ...` for media conversion because the environment supplies FFmpeg.

## Coding Style and Naming

Follow the surrounding Python style: four-space indentation, type hints where practical, `snake_case` for modules/functions/variables, and `PascalCase` for classes. Prefer `pathlib.Path`, explicit schemas, small helpers, and deterministic file outputs. No repository formatter or linter configuration is present, so keep changes consistent with nearby code and run focused tests after edits.

## Testing Guidelines

Add or update a corresponding `tests/test_<module>.py` test for behavior changes. Tests are primarily `unittest.TestCase` classes and should be deterministic, isolate temporary files with `TemporaryDirectory`, and avoid network or model downloads. There is no stated coverage threshold; run the full discovery command before submitting.

## Commits and Pull Requests

Use short, imperative commit subjects matching the existing history, such as `Add Auralis audio evaluation agent`. Pull requests should explain the behavior change, list validation commands and results, identify any data/model or Git LFS additions, and link the relevant issue or design note. Include screenshots only for documentation or generated-deck changes. Never commit API keys, `.env.local`, raw private evaluation data, or generated artifacts that belong in ignored directories.
