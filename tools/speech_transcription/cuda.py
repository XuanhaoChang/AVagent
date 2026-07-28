"""CUDA runtime environment helpers for pip-installed NVIDIA libraries."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


def cuda_library_dirs() -> tuple[Path, ...]:
    directories = []
    try:
        import nvidia.cublas.lib
        import nvidia.cudnn.lib
    except ImportError:
        return ()
    for module in (nvidia.cublas.lib, nvidia.cudnn.lib):
        module_file = getattr(module, "__file__", None)
        if module_file:
            directory = Path(module_file).resolve().parent
        else:
            module_paths = tuple(getattr(module, "__path__", ()))
            if not module_paths:
                continue
            directory = Path(module_paths[0]).resolve()
        if directory.is_dir() and directory not in directories:
            directories.append(directory)
    return tuple(directories)


def cuda_process_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(source or os.environ)
    directories = [str(path) for path in cuda_library_dirs()]
    existing = environment.get("LD_LIBRARY_PATH", "")
    if existing:
        directories.append(existing)
    environment["LD_LIBRARY_PATH"] = ":".join(directories)
    environment["AURALIS_CUDA_BOOTSTRAPPED"] = "1"
    return environment
