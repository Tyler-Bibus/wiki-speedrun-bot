"""Atomic, resumable training checkpoints and run artifact helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
import re
import shutil
import tempfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Training requires the optional train dependencies; install with `uv sync --extra train`.") from exc
    return torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_fingerprint(config: Any) -> str:
    values = asdict(config) if is_dataclass(config) else config
    encoded = json.dumps(values, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def file_fingerprints(paths: list[Path]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required training input files are missing: {', '.join(missing)}")
    return {str(path.resolve()): sha256_file(path) for path in paths}


def seed_everything(seed: int) -> None:
    """Set all training RNGs before model initialization on a new run."""
    random.seed(seed)
    try:
        import numpy
    except ImportError:
        pass
    else:
        numpy.random.seed(seed)
    torch = _torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def object_fingerprint(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def graph_fingerprint(graph: Any) -> str:
    return object_fingerprint({
        "articles": sorted(graph.articles),
        "links": {source: list(targets) for source, targets in sorted(graph.links.items())},
        "texts": dict(sorted(getattr(graph, "texts", {}).items())),
    })


def write_provenance(run_dir: Path, model: Any, data_hashes: dict[str, str],
                     selection: dict[str, Any]) -> None:
    import importlib.metadata
    import platform
    import subprocess

    try:
        git_revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, cwd=Path.cwd(),
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], check=True, capture_output=True,
            text=True, cwd=Path.cwd(),
        ).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        git_revision = None
        dirty = None
    torch = _torch()
    metadata = getattr(model, "metadata", {})
    package_versions = {}
    for package in ("torch", "transformers", "peft", "numpy", "kev"):
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None
    payload = {
        "schema_version": 1,
        "git_revision": git_revision,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "package_versions": package_versions,
        "uv_lock_sha256": sha256_file(Path("uv.lock")) if Path("uv.lock").is_file() else None,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "device": str(getattr(getattr(model, "model", None), "device", "unknown")),
        "model": metadata,
        "data_hashes": data_hashes,
        "selection": selection,
    }
    write_json_atomic(Path(run_dir) / "provenance.json", payload)


def resolve_checkpoint(value: str | Path) -> Path:
    """Resolve an update directory or latest/best JSON pointer."""
    path = Path(value).expanduser()
    if path.is_file() and path.suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        target = path.parent / payload["checkpoint"]
        if not target.is_dir():
            raise FileNotFoundError(f"Checkpoint pointer {path} refers to missing {target}")
        return target
    if path.is_dir() and not (path / "training_state.pt").is_file():
        for pointer_name in ("latest.json", "best.json"):
            pointer = path / pointer_name
            if pointer.is_file():
                payload = json.loads(pointer.read_text(encoding="utf-8"))
                target = path / payload["checkpoint"]
                if not target.is_dir():
                    raise FileNotFoundError(f"Checkpoint pointer {pointer} refers to missing {target}")
                return target
    return path


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def capture_rng_state() -> dict[str, Any]:
    torch = _torch()
    state = {
        "python": random.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    try:
        import numpy
    except ImportError:
        pass
    else:
        state["numpy"] = numpy.random.get_state()
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    torch = _torch()
    random.setstate(state["python"])
    torch.set_rng_state(state["torch_cpu"])
    if "numpy" in state:
        import numpy
        numpy.random.set_state(state["numpy"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def log_offsets(run_dir: Path) -> dict[str, int]:
    result = {}
    for name in ("metrics.jsonl", "transitions.jsonl", "episodes.jsonl"):
        path = run_dir / name
        result[name] = path.stat().st_size if path.exists() else 0
    return result


def truncate_logs(run_dir: Path, offsets: dict[str, int]) -> None:
    for name, offset in offsets.items():
        path = run_dir / name
        if not path.exists():
            if offset:
                raise ValueError(f"Checkpoint expects missing log {path}")
            continue
        if path.stat().st_size < offset:
            raise ValueError(f"Run log {path} is shorter than the committed checkpoint offset")
        with path.open("r+b") as stream:
            stream.truncate(int(offset))


def _checkpoint_update(directory: Path, run_id: str) -> int | None:
    """Return an update number only for a committed checkpoint of this run."""
    if directory.is_symlink() or not directory.is_dir():
        return None
    match = re.fullmatch(r"update-(\d+)", directory.name)
    if match is None:
        return None
    update = int(match.group(1))
    try:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if (not isinstance(manifest, dict) or manifest.get("run_id") != run_id
            or type(manifest.get("completed_update")) is not int
            or manifest["completed_update"] != update):
        return None
    return update


def _checkpoint_pointer_targets(root: Path, run_id: str) -> set[str] | None:
    """Read valid pointer targets; None means pruning must be skipped."""
    targets = set()
    for pointer_name in ("latest.json", "best.json"):
        pointer = root / pointer_name
        if pointer.is_symlink():
            return None
        if not pointer.exists():
            continue
        if not pointer.is_file():
            return None
        try:
            payload = json.loads(pointer.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        name = payload.get("checkpoint") if isinstance(payload, dict) else None
        if not isinstance(name, str) or Path(name).name != name:
            return None
        target = root / name
        if _checkpoint_update(target, run_id) is None:
            return None
        targets.add(name)
    return targets


def _keep_last_checkpoints(config: Any) -> int:
    save = config.get("save", {}) if isinstance(config, dict) else getattr(config, "save", None)
    value = save.get("keep_last_checkpoints", 2) if isinstance(save, dict) else getattr(
        save, "keep_last_checkpoints", 2
    )
    if type(value) is not int or value < 1:
        raise ValueError("save.keep_last_checkpoints must be a positive integer")
    return value


def _prune_checkpoints(root: Path, run_id: str, keep_last: int,
                       just_saved: Path) -> None:
    pointer_targets = _checkpoint_pointer_targets(root, run_id)
    if pointer_targets is None:
        return
    committed = []
    for child in root.iterdir():
        update = _checkpoint_update(child, run_id)
        if update is not None:
            committed.append((update, child))
    committed.sort(key=lambda item: item[0])
    protected = pointer_targets | {just_saved.name}
    protected.update(path.name for _, path in committed[-keep_last:])
    for _, directory in committed:
        if directory.name not in protected:
            shutil.rmtree(directory)


def save_training_checkpoint(*, model: Any, optimizer: Any, config: Any,
                             run_dir: Path, run_id: str, phase: str, update: int,
                             cursor: dict[str, Any], best_score: dict[str, Any] | None,
                             data_hashes: dict[str, str], make_latest: bool = True,
                             make_best: bool = False) -> Path:
    """Export model and local state into an atomically published directory."""
    torch = _torch()
    root = Path("checkpoints") / run_id
    root.mkdir(parents=True, exist_ok=True)
    name = f"update-{update:06d}"
    final_dir = root / name
    if final_dir.exists():
        raise FileExistsError(f"Refusing to overwrite committed checkpoint {final_dir}")
    temporary_dir = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=root))
    try:
        model.save_checkpoint(temporary_dir)
        training_state = {
            "schema_version": 1,
            "phase": phase,
            "completed_update": update,
            "cursor": cursor,
            "optimizer": optimizer.state_dict(),
            "rng": capture_rng_state(),
            "best_score": best_score,
            "run_id": run_id,
            "config_fingerprint": config_fingerprint(config),
            "data_hashes": data_hashes,
            "log_offsets": log_offsets(run_dir),
        }
        torch.save(training_state, temporary_dir / "training_state.pt")
        hashes = {}
        for path in sorted(temporary_dir.rglob("*")):
            if path.is_file():
                hashes[path.relative_to(temporary_dir).as_posix()] = sha256_file(path)
        manifest = {
            "schema_version": 1,
            "phase": phase,
            "run_id": run_id,
            "completed_update": update,
            "config_fingerprint": config_fingerprint(config),
            "data_hashes": data_hashes,
            "log_offsets": training_state["log_offsets"],
            "files": hashes,
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8", newline="\n",
        )
        os.replace(temporary_dir, final_dir)
        pointer = {"schema_version": 1, "checkpoint": name}
        if make_latest:
            write_json_atomic(root / "latest.json", pointer)
        if make_best:
            write_json_atomic(root / "best.json", pointer)
        _prune_checkpoints(root, run_id, _keep_last_checkpoints(config), final_dir)
        return final_dir
    except BaseException:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise


def load_training_state(checkpoint: str | Path, *, config: Any,
                        optimizer: Any | None = None,
                        expected_data_hashes: dict[str, str] | None = None,
                        run_dir: Path | None = None) -> dict[str, Any]:
    """Validate, restore optimizer/RNG, and trim logs to the committed boundary."""
    torch = _torch()
    directory = resolve_checkpoint(checkpoint)
    manifest_path = directory / "manifest.json"
    state_path = directory / "training_state.pt"
    if not manifest_path.is_file() or not state_path.is_file():
        raise ValueError(f"{directory} is not a resumable training checkpoint")
    validate_checkpoint_files(directory, config=config,
                              expected_data_hashes=expected_data_hashes)
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    # Loading the model must happen before this helper so RNG is restored only
    # after tokenizer/base initialization has consumed its startup randomness.
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if run_dir is not None:
        truncate_logs(run_dir, state["log_offsets"])
    restore_rng_state(state["rng"])
    return state


def validate_checkpoint_files(checkpoint: str | Path, *, config: Any,
                              expected_data_hashes: dict[str, str] | None = None) -> Path:
    """Verify exports and fingerprints before loading model or training state."""
    directory = resolve_checkpoint(checkpoint)
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"{directory} is missing checkpoint manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for relative_name, expected_hash in manifest.get("files", {}).items():
        artifact = directory / relative_name
        if not artifact.is_file() or sha256_file(artifact) != expected_hash:
            raise ValueError(f"Checkpoint artifact hash mismatch: {relative_name}")
    if manifest.get("config_fingerprint") != config_fingerprint(config):
        raise ValueError("Resume configuration does not match the checkpoint")
    if expected_data_hashes is not None and manifest.get("data_hashes") != expected_data_hashes:
        raise ValueError("Resume data files do not match the checkpoint")
    return directory


def make_run_dir(config: Any, phase: str) -> tuple[Path, str]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_id = f"{stamp}-{phase}"
    run_dir = Path(config.run_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    values = asdict(config)
    values["run_id"] = run_id
    values["schema_version"] = config.schema_version
    (run_dir / "config.json").write_text(
        json.dumps(values, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8", newline="\n",
    )
    for name in ("metrics.jsonl", "transitions.jsonl", "episodes.jsonl"):
        (run_dir / name).touch()
    return run_dir, run_id


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one finite JSON record and flush it to the run artifacts."""
    with Path(path).open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False, separators=(",", ":")))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
