# Architecture

AVAgent separates observations, tool execution state, and defect decisions so
an unavailable signal cannot silently become a negative finding.

## Execution flow

1. The visual agent extracts prompt requirements and inspects visual evidence.
2. Seed-Lite optionally reviews narrowly scoped logo, motion, and physical
   candidates.
3. Auralis collects audio, ASR, speaker, OCR, and subtitle-alignment evidence.
4. AVBench/SyncNet evaluates face-track audio-video synchronization in an
   isolated process.
5. The classic-check harness maps cached stage results into ten issue-oriented
   checks.
6. Final synthesis organizes and deduplicates the supported issue union without
   inventing evidence.

## Core contracts

- `EvaluationContext` contains only inference-safe sample fields and cached
  tool results.
- `ToolResult` records execution status, evidence, artifacts, diagnostics,
  errors, and usage separately.
- `ClassicCheckResult` records a check decision and the evidence keys used.
- `EvaluationResult` combines the ten checks, final issues, tool trace, and a
  compatibility log.

Statuses distinguish successful execution, failure, and non-applicability.
Missing audio, failed OCR, or an unavailable checkpoint is not evidence that a
video has no corresponding defect.

## Evidence boundaries

| Signal | Supports | Does not prove by itself |
| --- | --- | --- |
| Visual agent | Prompt requirements and visible candidates | Audio defects |
| ASR | Spoken words and approximate timing | Voice identity or lip sync |
| OCR | Candidate visible strings and boxes | That a shape is a subtitle |
| OCR visual gate | Subtitle/scene-text/logo classification | Unobserved issues |
| Speaker tracks | Relative voice consistency | Absolute identity without a reference |
| AVBench/SyncNet | Face-track timing evidence | A defect from low confidence alone |
| Final synthesis | Organization and deduplication | New unsupported findings |

## Dependency isolation

The parent process handles orchestration, OCR, and lightweight evidence. The
SenseVoice and AVBench worker runs in a separate environment containing the
PyTorch/CUDA stack. `agents/avbench_worker.py` imports the project adapter from
`av_eval.syncnet`; the `third_party/AVBench` checkout remains upstream-clean.

## Artifacts

Each run can produce a prediction CSV and JSONL trace. The trace includes stage
statistics, `classic_checks`, and `tool_results`. Generated artifacts are local
and ignored by Git; publish only small, intentionally anonymized examples.
