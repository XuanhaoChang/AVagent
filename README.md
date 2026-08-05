# AVagent

**Evidence-first evaluation for generated audio-video content.**

AVagent evaluates whether a generated video follows its prompt across the
dimensions that are easy to miss in a frame-only review: speech, subtitles,
speaker identity, lip-sync, text and logos, visual continuity, and basic
physical plausibility. It combines deterministic measurements with model-based
reasoning, while keeping the evidence chain visible for human review.

> The repository contains source code and tests only. Private datasets, model
> weights, API credentials, and generated evaluation artifacts are kept outside
> Git.

## What it does

```text
                         ┌─────────────────────────┐
                         │  Prompt + generated AV  │
                         └────────────┬────────────┘
                                      │
                 ┌────────────────────┼────────────────────┐
                 │                    │                    │
                 ▼                    ▼                    ▼
        ┌────────────────┐   ┌────────────────┐   ┌────────────────┐
        │ GPT-A planning │   │ Auralis        │   │ AVBench/SyncNet│
        │ prompt issues  │   │ ASR + OCR +    │   │ lip-sync       │
        │ and criteria   │   │ Gemini review  │   │ evidence       │
        └───────┬────────┘   └───────┬────────┘   └───────┬────────┘
                │                    │                    │
                └────────────────────┼────────────────────┘
                                     ▼
                         ┌─────────────────────────┐
                         │ Final GPT synthesis    │
                         │ merge, deduplicate,    │
                         │ classify, preserve     │
                         │ evidence boundaries   │
                         └────────────┬────────────┘
                                      ▼
                         ┌─────────────────────────┐
                         │ Prediction + run log   │
                         │ + human-review package  │
                         └─────────────────────────┘
```

The central design principle is simple: a model may propose an issue, but a
claim is only promoted when the available evidence supports it. ASR, OCR,
speaker tracks, timing measurements, and visual observations remain separate
so that a reviewer can trace a result back to its source.

## Evidence boundaries

| Signal | Used for | Not treated as proof by itself |
| --- | --- | --- |
| GPT-A | Extracting prompt requirements and expected entities | A visual or audio defect in the video |
| ASR and constrained candidate scoring | Spoken content, word substitutions, speaker turns | Ground-truth semantics without context |
| OCR | Visible text, subtitles, logos, and text mismatches | Audio transcription |
| Gemini / Auralis | Evidence-aware multimodal judgment and issue wording | A substitute for missing evidence |
| CAM++ / speaker tracks | Voice-to-entity consistency checks | Absolute speaker identity without prompt context |
| AVBench / SyncNet | Audio-video synchronization | A defect from a raw score or boundary hit alone |
| Seed-Lite | Targeted visual checks such as logos and motion continuity | A universal detector for every visual artifact |

## Quick start

The project is tested in the `avagent` Python environment.

```bash
conda activate avagent
python -m pip install -r requirements-audio-agent.txt
python -m unittest discover -s tests -p 'test_*.py'
```

For a dependency-free local smoke check of the media-evidence path:

```bash
python scripts/smoke_auralis_local.py /path/to/video.mp4
```

Keep credentials in the local environment (for example `.env.local`); never
place them in source files or commits.

## Run Agent-D

`call_ffmpeg_skill_gpt_d.py` is the current evaluation entry point. A typical
auditable run writes both the prediction CSV and a JSONL execution log into a
task-specific output directory:

```bash
python call_ffmpeg_skill_gpt_d.py \
  --input-csv output/captionErr/agentd_input.csv \
  --latentsync-root .external/LatentSync \
  --avbench-syncnet-ckpt .external/LatentSync/checkpoints/auxiliary/syncnet_v2.model \
  --avbench-python .conda-envs/avbench/bin/python \
  --output-csv output/captionErr/agentd/pred.csv \
  --run-log output/captionErr/agentd/run.jsonl
```

The input CSV should identify the media and prompt for each row. Validate row
IDs and media paths before a large run, and inspect `run.jsonl` for record
counts, failures, and audio-availability diagnostics.

## Human-review workflow

Review packages preserve the original sample structure and include the
Agent-D output plus deterministic ASR/OCR evidence. The standard flow is:

```bash
python scripts/export_human_review_samples.py \
  --input-csv output/<task>/agentd_input.csv \
  --output-root output/<task>/human_review_samples

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

The resulting package is designed to answer three questions independently:

1. What did the prompt require?
2. What does the generated video provide as evidence?
3. Did the evaluator describe and localize the defect accurately?

## Repository map

```text
av_eval/                 Dataset, taxonomy, routing, scoring, and CLI modules
agents/auralis/          ASR/OCR, constrained evidence, and Gemini orchestration
tools/                   Shared media, transcription, subtitle, and utility tools
scripts/                 Experiment, export, review, and smoke-test entry points
tests/                   Deterministic unittest coverage
third_party/AVBench/     AVBench submodule for synchronization evaluation
call_ffmpeg_skill_gpt_d.py  Current Agent-D entry point
call_ffmpeg_skill.py        Original baseline for comparison
```

Local `input/`, `output/`, `models/`, checkpoints, caches, and generated media
are intentionally ignored. Obtain private assets through the approved local
deployment process instead of adding them to this repository.

## Development

Use small, deterministic changes and add a corresponding test for behavior
changes. The project uses `unittest` as its validation interface:

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m av_eval.cli capacity-plan --output /tmp/avagent-capacity.json
```

Before submitting a change, check that no credentials, raw evaluation media,
model weights, or generated outputs are staged:

```bash
git status --short
git diff --cached --name-only
```

## License and data

No private evaluation data or model distribution is implied by this source
repository. Follow the licenses and access terms of each external dependency
and obtain any required checkpoints or datasets separately.
