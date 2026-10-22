"""Supervised cross-entropy updates for precomputed tournament nodes."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from ..data.load import load_graph
from ..data.tasks import Task
from .checkpoints import (
    append_jsonl,
    file_fingerprints,
    load_training_state,
    make_run_dir,
    graph_fingerprint,
    object_fingerprint,
    resolve_checkpoint,
    save_training_checkpoint,
    seed_everything,
    validate_checkpoint_files,
    write_provenance,
    write_json_atomic,
)
from .rollouts import autocast_context


def _load_model(checkpoint: str | Path, config: Any):
    from ..models.trainable_kev import TrainableKev

    source = resolve_checkpoint(checkpoint) if Path(checkpoint).expanduser().exists() else str(checkpoint)
    return TrainableKev.from_checkpoint(str(source), config.model.device, context=config.choices)


def _jsonl_offsets(path: Path, limit: int | None = None) -> list[int]:
    offsets = []
    with path.open("r", encoding="utf-8") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object in {path} at byte offset {offset}")
            offsets.append(offset)
            if limit is not None and len(offsets) >= limit:
                break
    return offsets


def _read_record(path: Path, offset: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        stream.seek(offset)
        value = json.loads(stream.readline())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path} at byte offset {offset}")
    return value


def _sft_loss(model: Any, record: dict[str, Any], config: Any):
    import torch
    import torch.nn.functional as functional

    candidate_ids = record["candidate_ids"]
    label = record["label"]
    if label not in candidate_ids:
        raise ValueError(f"SFT label {label!r} is absent from candidate_ids")
    expected = list(record["request"]["questions"]["next_link"]["criteria"])
    if expected != candidate_ids:
        raise ValueError("SFT request choice order does not match candidate_ids")
    with autocast_context(config):
        encoded = model.encode(record["request"])
        logits = model.logits(encoded).float()
        if logits.ndim != 1 or logits.numel() != len(candidate_ids):
            raise ValueError("SFT logits do not match candidate_ids")
        target = torch.tensor([candidate_ids.index(label)], device=logits.device)
        return functional.cross_entropy(logits.unsqueeze(0), target)


def _development_score(model: Any, graph: Any, config: Any,
                       dev_tasks: list[Task]) -> tuple[dict[str, Any], dict[str, Any]]:
    from .rl import _evaluate

    metrics = _evaluate(model, graph, dev_tasks, config)
    score = {
        "success_rate": metrics["success_rate"],
        "mean_success_steps": metrics["mean_success_steps"],
    }
    return metrics, score


def _improves(candidate: dict[str, Any], best: dict[str, Any] | None) -> bool:
    if best is None or candidate["success_rate"] > best["success_rate"]:
        return True
    candidate_steps = candidate["mean_success_steps"]
    best_steps = best["mean_success_steps"]
    candidate_steps = candidate_steps if candidate_steps is not None else float("inf")
    best_steps = best_steps if best_steps is not None else float("inf")
    return candidate["success_rate"] == best["success_rate"] and candidate_steps < best_steps


def train_sft(config: Any, *, resume: str | Path | None = None,
               max_updates: int | None = None, max_examples: int | None = None,
               max_tasks: int | None = None,
               graph: Any | None = None, model: Any | None = None) -> Path:
    """Train on SFT JSONL and select epochs using deterministic dev navigation.

    ``max_updates`` is a total optimizer-step ceiling, including resumed work.
    Injected ``model`` and ``graph`` keep small CPU smoke runs independent of Kev.
    """
    import torch

    if max_updates is None:
        max_updates = config.sft.max_updates
    if max_updates is not None and max_updates < 1:
        raise ValueError("max_updates must be positive")
    if max_examples is not None and max_examples < 1:
        raise ValueError("max_examples must be positive")
    if max_tasks is not None and max_tasks < 1:
        raise ValueError("max_tasks must be positive")
    source = resume or config.model.checkpoint
    if resume is None:
        seed_everything(config.seed)
    injected_dependencies = model is not None or graph is not None
    if graph is None:
        graph = load_graph(Path(config.data_dir))
    train_path = Path(config.data.sft_train)
    dev_path = Path(config.data.dev_tasks)
    fingerprint_inputs = [train_path, Path(config.data.sft_dev), dev_path,
                          Path(config.data.split_manifest)]
    data_hashes = file_fingerprints(fingerprint_inputs)
    offsets = _jsonl_offsets(train_path, max_examples)
    if not offsets:
        raise ValueError(f"SFT training data is empty: {train_path}")
    from .rl import _development_tasks
    dev_tasks = _development_tasks(config, max_tasks)
    selection = {
        "max_examples": max_examples,
        "max_tasks": max_tasks,
        "dev_task_budget": (max_tasks if max_tasks is not None
                            else config.save.evaluation_max_tasks),
        "dev_max_tasks": len(dev_tasks),
        "dev_tasks": [{"start": task.start, "target": task.target} for task in dev_tasks],
    }
    data_hashes["graph:sha256"] = graph_fingerprint(graph)
    data_hashes["selection:sha256"] = object_fingerprint(selection)
    prepared_manifest = None
    if not injected_dependencies:
        from ..data.training import validate_prepared_data
        prepared_manifest = validate_prepared_data(config)

    checkpoint_dir = None
    if resume is not None:
        checkpoint_dir = validate_checkpoint_files(
            resume, config=config, expected_data_hashes=data_hashes,
        )
    model = model or _load_model(checkpoint_dir or source, config)
    if prepared_manifest is not None:
        validate_prepared_data(config, model=model)
    model.train(True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.sft.learning_rate)

    if resume is None:
        run_dir, run_id = make_run_dir(config, "sft")
        write_provenance(run_dir, model, data_hashes, selection)
        update = 0
        best_score = None
        cursor = {"epoch": 0, "offset": 0, "order": None}
    else:
        assert checkpoint_dir is not None
        last_checkpoint = checkpoint_dir
        preview = torch.load(checkpoint_dir / "training_state.pt", map_location="cpu", weights_only=False)
        run_id = preview["run_id"]
        run_dir = Path(config.run_dir) / run_id
        state = load_training_state(checkpoint_dir, config=config, optimizer=optimizer,
                                    expected_data_hashes=data_hashes, run_dir=run_dir)
        if state["phase"] != "sft":
            raise ValueError("Cannot resume SFT from a non-SFT checkpoint")
        update = int(state["completed_update"])
        cursor = state["cursor"]
        best_score = state.get("best_score")

    update_limit_reached = max_updates is not None and update >= max_updates
    if int(cursor["epoch"]) >= config.sft.epochs or update_limit_reached:
        return checkpoint_dir

    started = time.perf_counter()
    last_checkpoint = checkpoint_dir
    optimizer.zero_grad(set_to_none=True)
    try:
        while int(cursor["epoch"]) < config.sft.epochs:
            if max_updates is not None and update >= max_updates:
                break

            epoch = int(cursor["epoch"])
            if cursor["order"] is None:
                order = list(range(len(offsets)))
                random.shuffle(order)
                cursor["order"] = order
                cursor["offset"] = 0
            else:
                order = list(cursor["order"])

            start = int(cursor["offset"])
            end = min(start + config.sft.gradient_accumulation, len(order))
            batch_indices = order[start:end]
            batch_size = len(batch_indices)
            if batch_size == 0:
                raise RuntimeError("SFT cursor reached an empty batch")

            optimizer.zero_grad(set_to_none=True)
            total_loss = 0.0
            for index in batch_indices:
                loss = _sft_loss(model, _read_record(train_path, offsets[index]), config)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite SFT loss at update {update + 1}")
                (loss / batch_size).backward()
                total_loss += float(loss.detach().item())

            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite SFT gradient at update {update + 1}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1

            cursor["offset"] = end
            epoch_finished = end >= len(order)
            if epoch_finished:
                cursor = {"epoch": epoch + 1, "offset": 0, "order": None}

            append_jsonl(run_dir / "metrics.jsonl", {
                "schema_version": 1, "phase": "sft", "update": update,
                "epoch": epoch, "loss": total_loss / batch_size,
                "gradient_norm": float(gradient_norm.detach().item()),
                "elapsed_seconds": time.perf_counter() - started,
            })
            print(f"SFT update {update}: loss={total_loss / batch_size:.6f}", flush=True)

            final_update = (
                int(cursor["epoch"]) >= config.sft.epochs or
                (max_updates is not None and update >= max_updates)
            )
            should_evaluate = (
                update % config.save.evaluate_every_updates == 0 or
                epoch_finished or final_update
            )
            improved = False
            if should_evaluate:
                metrics, score = _development_score(model, graph, config, dev_tasks)
                model.train(True)
                improved = _improves(score, best_score)
                if improved:
                    best_score = score
                append_jsonl(run_dir / "metrics.jsonl", {
                    "schema_version": 1, "phase": "sft", "event": "evaluation",
                    "update": update,
                    **{key: value for key, value in metrics.items() if key != "episodes"},
                })
                write_json_atomic(run_dir / "metrics.json", {
                    key: value for key, value in metrics.items() if key != "episodes"
                })

            if (update % config.save.checkpoint_every_updates == 0 or
                    improved or final_update):
                last_checkpoint = save_training_checkpoint(
                    model=model, optimizer=optimizer, config=config, run_dir=run_dir,
                    run_id=run_id, phase="sft", update=update, cursor=cursor,
                    best_score=best_score, data_hashes=data_hashes,
                    make_best=improved,
                )
                print(f"SFT checkpoint: {last_checkpoint}", flush=True)

        if last_checkpoint is None:
            raise RuntimeError("SFT run completed without a checkpoint")
        return last_checkpoint
    except BaseException as exc:
        write_json_atomic(run_dir / "error.json", {
            "schema_version": 1, "phase": "sft", "error_type": type(exc).__name__,
            "message": str(exc), "update": update,
            "last_checkpoint": str(last_checkpoint) if last_checkpoint else None,
        })
        raise
