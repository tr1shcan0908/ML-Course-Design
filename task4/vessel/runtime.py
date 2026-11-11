"""Run metadata, atomic checkpoints, reproducible random states."""

from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import shutil
import sys

import numpy as np
import torch


def now() -> str:
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds")


def write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.use_deterministic_algorithms(deterministic)


def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(states: dict) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch"])
    if states["cuda"]:
        torch.cuda.set_rng_state_all(states["cuda"])


def source_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def record_environment(directory: Path, root: Path) -> dict:
    packages = {}
    for name in ("torch", "numpy", "Pillow", "filelock", "fsspec", "networkx", "Jinja2", "MarkupSafe", "sympy", "mpmath", "typing_extensions", "setuptools"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    environment = {
        "created_at": now(), "timezone": "Asia/Shanghai", "python": sys.version,
        "python_executable": sys.executable, "platform": platform.platform(),
        "cuda_build": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
        "packages": packages, "source_sha256": source_digest(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "command": sys.argv,
    }
    write_json(directory / "environment.json", environment)
    lock_lines = ["# Runtime used for this experiment; Python 3.11, Windows, NVIDIA GPU.",
                  "--extra-index-url https://download.pytorch.org/whl/cu130"]
    lock_lines.extend(f"{name}=={version}" for name, version in packages.items())
    (directory / "requirements.lock.txt").write_text("\n".join(lock_lines) + "\n", encoding="utf-8")
    snapshot = directory / "source" / "vessel"
    snapshot.mkdir(parents=True, exist_ok=True)
    for path in Path(__file__).parent.glob("*.py"):
        shutil.copy2(path, snapshot / path.name)
    for name in ("folds_v1.json", "dataset_manifest.json"):
        shutil.copy2(root / "experiments" / name, directory / name)
    return environment
