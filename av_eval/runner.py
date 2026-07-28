"""Build credential-free commands for remote experiment execution."""

from __future__ import annotations

from pathlib import Path


def build_experiment_commands(
    *,
    python: str,
    script: Path,
    models: dict[str, str],
    profiles: tuple[str, ...],
    limit: int,
    start: int,
    output_root: Path,
) -> list[list[str]]:
    commands: list[list[str]] = []
    for label, model in models.items():
        for profile in profiles:
            run_dir = output_root / label / profile
            commands.append(
                [
                    python,
                    str(script),
                    "--model",
                    model,
                    "--profile",
                    profile,
                    "--start",
                    str(start),
                    "--limit",
                    str(limit),
                    "--output-csv",
                    str(run_dir / "pred.csv"),
                    "--run-log",
                    str(run_dir / "run.jsonl"),
                ]
            )
    return commands


def build_capacity_commands(
    *,
    python: str,
    script: Path,
    model: str,
    sample_index: int,
    image_counts: tuple[int, ...],
    output_root: Path,
    input_csv: Path | None = None,
) -> list[list[str]]:
    commands: list[list[str]] = []
    for count in image_counts:
        run_dir = output_root / f"images_{count:03d}"
        command = [
                python,
                str(script),
                "--model",
                model,
                "--profile",
                "baseline_a",
                "--start",
                str(sample_index),
                "--limit",
                "1",
                "--video-frame-fps",
                "12",
                "--max-video-frames",
                str(count),
                "--output-csv",
                str(run_dir / "pred.csv"),
                "--run-log",
                str(run_dir / "run.jsonl"),
            ]
        if input_csv is not None:
            command.extend(["--input-csv", str(input_csv)])
        commands.append(command)
    return commands
