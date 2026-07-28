"""Load project-local environment variables without external dependencies."""

from __future__ import annotations

import os
import shlex
from collections.abc import MutableMapping
from pathlib import Path


def load_project_env(
    path: Path,
    *,
    environ: MutableMapping[str, str] | None = None,
    override: bool = False,
) -> None:
    target = os.environ if environ is None else environ
    if not path.is_file():
        return
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"{path}:{line_number} 不是 KEY=VALUE 格式")
        name, raw_value = line.split("=", 1)
        name = name.strip()
        if not name or not name.replace("_", "").isalnum() or name[0].isdigit():
            raise ValueError(f"{path}:{line_number} 环境变量名无效")
        parts = shlex.split(raw_value, comments=True, posix=True)
        value = " ".join(parts) if parts else ""
        if override or name not in target:
            target[name] = value
