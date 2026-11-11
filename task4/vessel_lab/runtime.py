"""Extended provenance without changing the frozen v1 package."""

import hashlib
import importlib.metadata
from pathlib import Path
import shutil

from vessel.runtime import (capture_rng, restore_rng, now, write_json,
                            save_checkpoint, seed_everything)
from vessel.runtime import record_environment as record_base_environment


def source_files():
    root = Path(__file__).resolve().parents[1]
    files = list((root / "vessel").glob("*.py"))
    files += [root / "vessel_lab" / name for name in ("__init__.py", "runtime.py", "train.py", "models.py", "data.py")]
    files += list((root / "vessel_lab/vendor").glob("*.py"))
    return sorted(files)


def source_digest():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in source_files():
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def record_environment(directory: Path, root: Path):
    record = record_base_environment(directory, root)
    record["base_vessel_source_sha256"] = record["source_sha256"]
    record["source_sha256"] = source_digest()
    record["packages"]["torchvision"] = importlib.metadata.version("torchvision")
    write_json(directory / "environment.json", record)
    for path in source_files():
        destination = directory / "source" / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    for name in ("LICENSE", "README.md"):
        shutil.copy2(root / "vessel_lab/vendor" / name, directory / "source/vessel_lab/vendor" / name)
    with (directory / "requirements.lock.txt").open("a", encoding="utf-8") as stream:
        stream.write(f"torchvision=={record['packages']['torchvision']}\n")
    return record
