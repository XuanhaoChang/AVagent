# Human Review Sample Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a standalone exporter that creates 100 human-review folders containing copied media and multiline GT/prediction JSON files.

**Architecture:** A focused `av_eval.review_export` module will parse and validate aligned CSV rows, resolve local media through the existing legacy-path resolver, and generate a complete temporary directory before atomically renaming it. A thin CLI script supplies the fixed five prediction inputs and user-overridable paths.

**Tech Stack:** Python 3.10 standard library, unittest, existing `av_eval.data` helpers.

---

### Task 1: Export one sample with readable JSON

**Files:**
- Create: `av_eval/review_export.py`
- Create: `tests/test_review_export.py`

- [x] **Step 1: Write the failing test**

Create a temporary GT CSV, five aligned prediction CSVs, one video and two references. Call:

```python
export_review_samples(
    gt_csv=gt_csv,
    prediction_csvs=prediction_csvs,
    media_root=media_root,
    output_root=output_root,
)
```

Assert that `sample_001` contains copied media, `input.json`, `gt.json`, and all five prediction files. Assert every JSON file contains `\n  {` or nested multiline indentation, ends in `\n`, and round-trips through `json.loads`.

- [x] **Step 2: Run test to verify it fails**

Run:

```bash
conda run -n avagent python -m unittest tests.test_review_export -v
```

Expected: fail with `ModuleNotFoundError: No module named 'av_eval.review_export'`.

- [x] **Step 3: Implement minimal exporter**

Implement:

```python
PREDICTION_FILES = {
    "gpt_a": Path("output/benchmark/runs/gpt/baseline_a/pred.csv"),
    "gpt_b": Path("output/benchmark/runs/gpt/harness_b/pred.csv"),
    "seed_a": Path("output/benchmark/runs/seed_lite/baseline_a/pred.csv"),
    "seed_b": Path("output/benchmark/runs/seed_lite/harness_b/pred.csv"),
    "seed_c": Path("output/benchmark/runs/seed_lite/harness_c/pred.csv"),
}
```

Use `csv.DictReader`, `extract_gold_array`, `resolve_legacy_media_path`, `json.loads`, `json.dumps(..., ensure_ascii=False, indent=2) + "\n"`, and `shutil.copy2`. Validate exact sample-ID alignment and JSON array-of-object structure. Generate under a sibling temporary directory and rename only after all rows succeed.

- [x] **Step 4: Run focused test**

Run:

```bash
conda run -n avagent python -m unittest tests.test_review_export -v
```

Expected: all review-export tests pass.

### Task 2: CLI and full 100-sample generation

**Files:**
- Create: `scripts/export_human_review_samples.py`
- Modify: `tests/test_review_export.py`

- [x] **Step 1: Add a failing CLI-default test**

Assert the parser defaults point to `input/gt.csv`, the five approved prediction CSVs, `input`, and `output/human_review_samples`.

- [x] **Step 2: Implement the CLI**

Create a thin `main()` that parses:

```text
--gt-csv
--media-root
--output-root
--gpt-a
--gpt-b
--seed-a
--seed-b
--seed-c
```

Call `export_review_samples` and print only the final sample count and output directory.

- [x] **Step 3: Run focused and full tests**

Run:

```bash
conda run -n avagent python -m unittest tests.test_review_export -v
conda run -n avagent python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [x] **Step 4: Generate and verify the real review package**

Run:

```bash
conda run -n avagent python scripts/export_human_review_samples.py
```

Verify exactly 100 sample directories, every directory contains `input.json`, `gt.json`, five prediction JSON files, one video and the expected number of references, and all JSON files contain physical newlines and parse successfully.
