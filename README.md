# AVAgent

Evidence-first evaluation for generated audio-video content.

AVAgent checks whether generated videos follow prompts and reference images
across visual content, speech, subtitles, speaker consistency, and lip sync.
Its central rule is that model suggestions are candidates: only findings backed
by the available media evidence enter the final issue set.

> This repository contains source code and tests only. Datasets, model weights,
> credentials, generated media, and experiment outputs are intentionally kept
> outside Git.

## Highlights

- One auditable pipeline for visual inspection, ASR/OCR evidence, speaker
  analysis, targeted visual checks, and AVBench/SyncNet.
- Ten issue-oriented checks sharing cached tool results through a typed
  `EvaluationContext`.
- Evidence gates that distinguish `not_evaluable` from “no defect.”
- CSV predictions plus JSONL traces suitable for human review and regression
  analysis.
- Deterministic unit tests that do not download models or call remote APIs.

## Pipeline

```text
Prompt + references + generated video
                 │
       ┌─────────┼──────────┐
       ▼         ▼          ▼
 visual agent  Auralis   AVBench/SyncNet
              ASR/OCR      lip sync
       └─────────┼──────────┘
                 ▼
       shared evidence + classic checks
                 ▼
       final merge and deduplication
                 ▼
       prediction CSV + JSONL audit log
```

See [Architecture](docs/architecture.md) for component boundaries and evidence
contracts.

## Quick start

Clone the submodule and run the deterministic test suite:

```bash
git clone --recurse-submodules https://github.com/XuanhaoChang/AVagent.git
cd AVagent
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -p 'test_*.py'
```

Runtime components have separate dependency boundaries:

```bash
# Parent process: OCR and tabular runtime
python -m pip install -r requirements-audio-agent.txt

# Isolated AVBench/SenseVoice environment
python -m pip install -r third_party/AVBench/requirements.txt
python -m pip install -r requirements-avbench.txt
python -m pip install -r requirements-sensevoice.txt
```

The model-heavy environment also needs a compatible PyTorch/CUDA installation,
LatentSync, and the SyncNet/S3FD checkpoints. AVAgent never downloads these
assets as part of tests or repository setup.

## Configuration

Copy the template and fill it locally:

```bash
cp .env.example .env.local
```

AVAgent expects an OpenAI-compatible Chat Completions gateway capable of the
configured multimodal requests. Keep all credentials in `.env.local`; the file
is ignored by Git.

Required settings:

- `AVAGENT_API_KEY`
- `AVAGENT_API_URL`
- `AVAGENT_VISUAL_MODEL`

Optional specialist and local-runtime settings are documented in
[`.env.example`](.env.example).

## Run an evaluation

The public entry point is `run_avagent.py`. The input schema is demonstrated in
[`examples/input.example.csv`](examples/input.example.csv).

```bash
python run_avagent.py \
  --input-csv input/gt.csv \
  --output-csv output/demo/predictions.csv \
  --run-log output/demo/run.jsonl \
  --latentsync-root /path/to/LatentSync \
  --avbench-syncnet-ckpt /path/to/syncnet_v2.model \
  --avbench-python /path/to/avbench/python
```

Before a large run, validate row IDs and media paths with a one-row invocation:

```bash
python run_avagent.py --input-csv input/gt.csv --limit 1 \
  --output-csv output/smoke/predictions.csv \
  --run-log output/smoke/run.jsonl
```

The current benchmark-compatible CSV keeps the original six columns and adds
`GPT预测结果`. The `思考过程及标准答案` column is preserved in the output but is
never included in model requests.

## Human review

```bash
python scripts/export_human_review_samples.py \
  --input-csv input/gt.csv \
  --output-root output/demo/review_samples

python scripts/attach_prediction_to_review_samples.py \
  --prediction-csv output/demo/predictions.csv \
  --samples-root output/demo/review_samples \
  --label avagent \
  --run-log output/demo/run.jsonl

python scripts/review_text_predictions.py \
  --input-root output/demo/review_samples \
  --output-jsonl output/demo/review/results.jsonl \
  --summary-json output/demo/review/summary.json \
  --prediction-source avagent
```

## Repository layout

```text
agents/                  Specialist orchestration and evidence gates
agents/classic_checks/   Shared contracts and ten issue checks
av_eval/                 Dataset, audit, routing, review, and CLI modules
configs/                 Runtime prompts and non-secret configuration
docs/                    Public architecture documentation
examples/                Small, non-private input examples
scripts/                 Evaluation, export, review, and smoke commands
tests/                   Deterministic unittest suite
third_party/AVBench/     Upstream AVBench Git submodule
tools/                   Media, transcription, subtitle, and alignment tools
run_avagent.py           Main evaluation entry point
run_visual_baseline.py   Standalone visual-only baseline
```

Local datasets, outputs, checkpoints, caches, and archived experiments belong
under ignored paths such as `input/`, `output/`, `models/`, and `.local/`.

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and pull-request checks.
Report vulnerabilities using [SECURITY.md](SECURITY.md), not a public issue.

## License

Licensed under the [Apache License 2.0](LICENSE).
