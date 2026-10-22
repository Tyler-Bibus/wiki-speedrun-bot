"""On-policy REINFORCE training and deterministic navigation evaluation."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any, Callable

from ..choices import TournamentPolicy
from ..data.load import load_graph
from ..data.tasks import Task
from ..environment import NavigationEnvironment
from ..evaluation import summarize
from .checkpoints import (
    append_jsonl,
    file_fingerprints,
    graph_fingerprint,
    load_training_state,
    make_run_dir,
    object_fingerprint,
    resolve_checkpoint,
    save_training_checkpoint,
    seed_everything,
    validate_checkpoint_files,
    write_json_atomic,
    write_provenance,
)
from .rewards import reward_to_go
from .rollouts import autocast_context, collect_episode, replay_node


def _load_model(checkpoint: str | Path, config: Any):
    from ..models.trainable_kev import TrainableKev

    expanded = Path(checkpoint).expanduser()
    source = resolve_checkpoint(expanded) if expanded.exists() else str(checkpoint)
    return TrainableKev.from_checkpoint(str(source), config.model.device, context=config.choices)


def _task_path(config: Any, split: str) -> Path:
    if split == "dev":
        return Path(config.data.dev_tasks)
    if split == "test":
        return Path(config.data.test_tasks)
    if split == "train":
        return Path(config.data.rl_train_tasks)
    if split == "baseline":
        return Path("examples/baseline_tasks.json")
    raise ValueError("split must be dev, test, train, or baseline")


def _load_tasks(path: Path, limit: int | None = None) -> list[Task]:
    values = json.loads(path.read_text(encoding="utf-8"))
    tasks = [Task(start=str(item["start"]), target=str(item["target"])) for item in values]
    if limit is not None:
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError(f"Task file is empty: {path}")
    return tasks


def _development_tasks(config: Any, max_tasks: int | None = None) -> list[Task]:
    """Choose a fixed dev subset without consuming the training RNG stream."""
    tasks = _load_tasks(Path(config.data.dev_tasks))
    limit = max_tasks
    if limit is None:
        limit = getattr(config.save, "evaluation_max_tasks", 32)
    if limit is None or limit >= len(tasks):
        return tasks
    if limit < 1:
        raise ValueError("development evaluation limit must be positive")
    return random.Random(config.seed).sample(tasks, limit)


def _evaluate(
    model: Any,
    graph: Any,
    tasks: list[Task],
    config: Any,
    *,
    on_task_start: Callable[[Task, int], None] | None = None,
    on_transition: Callable[[dict[str, Any]], None] | None = None,
    on_episode: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    import torch

    underlying_model = getattr(model, "model", None)
    prior_mode = getattr(model, "training", getattr(underlying_model, "training", False))
    episodes = []
    try:
        model.eval()
        policy = TournamentPolicy(model, config.choices, sample=False)
        with torch.no_grad():
            for index, task in enumerate(tasks):
                if on_task_start is not None:
                    on_task_start(task, index)
                env = NavigationEnvironment(graph, task.start, task.target, config.rl.max_steps)
                decisions = []
                episode_id = f"eval-{index:06d}"
                while not env.done:
                    source = env.current
                    history = list(env.history)
                    with autocast_context(config):
                        choice = policy.choose(env)
                    env.step(choice["article"])
                    reason = env.termination_reason
                    nodes = choice.get("nodes", [])
                    decisions.append({"latency_ms": choice.get("latency_ms", 0.0),
                                      "nodes": len(nodes)})
                    if on_transition is not None:
                        on_transition({
                            "schema_version": 1,
                            "episode_id": episode_id,
                            "pair": {"start": task.start, "target": task.target},
                            "step": env.steps - 1,
                            "source": source,
                            "target": task.target,
                            "history": history,
                            "selected_link_id": choice["selected_link_id"],
                            "selected_article": choice["article"],
                            "next_article": env.current,
                            "termination_reason": reason,
                            "nodes": nodes,
                            "model_calls": len(nodes),
                            "latency_ms": float(choice.get("latency_ms", 0.0)),
                        })
                episode = {
                    "schema_version": 1,
                    "episode_id": episode_id,
                    "pair": {"start": task.start, "target": task.target},
                    "start": task.start,
                    "target": task.target,
                    "path": list(env.history),
                    "steps": env.steps,
                    "success": env.success,
                    "termination_reason": env.termination_reason,
                    "model_calls": sum(decision["nodes"] for decision in decisions),
                    "decisions": decisions,
                }
                episodes.append(episode)
                if on_episode is not None:
                    on_episode(episode)
    finally:
        if prior_mode:
            model.train(True)
    metrics = summarize(episodes, graph)
    successes = [episode["steps"] for episode in episodes if episode["success"]]
    metrics.update({
        "schema_version": 1,
        "failure_counts": {
            reason: sum(episode["termination_reason"] == reason for episode in episodes)
            for reason in ("step_limit", "dead_end")
        },
        "mean_success_steps": sum(successes) / len(successes) if successes else None,
        "total_model_calls": sum(episode["model_calls"] for episode in episodes),
        "mean_model_calls": (
            sum(episode["model_calls"] for episode in episodes) / len(episodes)
            if episodes else 0.0
        ),
        "episodes": episodes,
    })
    return metrics


def evaluate_trained(config: Any, checkpoint: str | Path, split: str = "dev",
                     max_tasks: int | None = None, graph: Any | None = None,
                     model: Any | None = None) -> dict[str, Any]:
    """Evaluate a trained checkpoint with deterministic tournament argmax."""
    if max_tasks is not None and max_tasks < 1:
        raise ValueError("max_tasks must be positive")
    model = model or _load_model(checkpoint, config)
    if graph is None:
        graph = load_graph(Path(config.data_dir))
    task_path = _task_path(config, split)
    tasks = _load_tasks(task_path, max_tasks)
    selection = {"split": split, "max_tasks": max_tasks}
    data_hashes = file_fingerprints([task_path])
    data_hashes["graph:sha256"] = graph_fingerprint(graph)
    data_hashes["selection:sha256"] = object_fingerprint(selection)
    run_dir, run_id = make_run_dir(config, "evaluate")
    write_provenance(run_dir, model, data_hashes, selection)

    progress: dict[str, Any] = {
        "task_index": None,
        "task": None,
        "last_completed_task_index": None,
    }

    def record_task_start(task: Task, index: int) -> None:
        progress["task_index"] = index
        progress["task"] = {"start": task.start, "target": task.target}

    def record_transition(transition: dict[str, Any]) -> None:
        append_jsonl(run_dir / "transitions.jsonl", {
            **transition,
            "policy_version": str(checkpoint),
        })

    def record_episode(episode: dict[str, Any]) -> None:
        append_jsonl(run_dir / "episodes.jsonl", {
            **episode,
            "policy_version": str(checkpoint),
        })
        progress["last_completed_task_index"] = progress["task_index"]

    try:
        metrics = _evaluate(
            model, graph, tasks, config,
            on_task_start=record_task_start,
            on_transition=record_transition,
            on_episode=record_episode,
        )
    except BaseException as error:
        write_json_atomic(run_dir / "error.json", {
            "schema_version": 1,
            "phase": "evaluate",
            "checkpoint": str(checkpoint),
            "split": split,
            "task_index": progress["task_index"],
            "task": progress["task"],
            "last_completed_task_index": progress["last_completed_task_index"],
            "error_type": type(error).__name__,
            "message": str(error),
        })
        raise
    metrics["split"] = split
    metrics["checkpoint"] = str(checkpoint)
    metrics["run_dir"] = str(run_dir)
    saved_metrics = {key: value for key, value in metrics.items() if key != "episodes"}
    append_jsonl(run_dir / "metrics.jsonl", {
        "schema_version": 1, "phase": "evaluate", "run_id": run_id,
        **saved_metrics,
    })
    write_json_atomic(run_dir / "metrics.json", saved_metrics)
    return metrics


def _episode_metrics(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    returns = [float(item["return"]) for item in episodes]
    successes = sum(bool(item["success"]) for item in episodes)
    steps = [int(item["steps"]) for item in episodes]
    node_count = sum(len(transition["nodes"]) for episode in episodes
                     for transition in episode["transitions"])
    entropy_values = [float(node["entropy"]) for episode in episodes
                      for transition in episode["transitions"] for node in transition["nodes"]]
    return {
        "episode_count": len(episodes),
        "mean_return": sum(returns) / len(returns) if returns else 0.0,
        "success_rate": successes / len(episodes) if episodes else 0.0,
        "mean_clicks": sum(steps) / len(steps) if steps else 0.0,
        "node_count": node_count,
        "mean_node_entropy": sum(entropy_values) / len(entropy_values) if entropy_values else 0.0,
    }


def train_rl(config: Any, *, init_from: str | Path | None = None,
             resume: str | Path | None = None, max_updates: int | None = None,
             max_tasks: int | None = None, graph: Any | None = None,
             model: Any | None = None) -> Path:
    """Run one fresh on-policy batch per update with node-level credit assignment."""
    import torch

    if max_updates is not None and max_updates < 1:
        raise ValueError("max_updates must be positive")
    if max_tasks is not None and max_tasks < 1:
        raise ValueError("max_tasks must be positive")
    source = resume or init_from or config.model.checkpoint
    if resume is None:
        seed_everything(config.seed)
    injected_dependencies = model is not None or graph is not None
    if graph is None:
        graph = load_graph(Path(config.data_dir))
    train_path = Path(config.data.rl_train_tasks)
    dev_path = Path(config.data.dev_tasks)
    manifest_path = Path(config.data.split_manifest)
    data_hashes = file_fingerprints([train_path, dev_path, manifest_path])
    train_tasks = _load_tasks(train_path, max_tasks)
    if len(train_tasks) < config.rl.task_pairs_per_update:
        raise ValueError("RL task file has fewer pairs than task_pairs_per_update")
    dev_tasks = _development_tasks(config, max_tasks)
    selection = {
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
    model.eval()  # dropout stays off for collection and replay
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.rl.learning_rate)

    if resume is None:
        run_dir, run_id = make_run_dir(config, "rl")
        write_provenance(run_dir, model, data_hashes, selection)
        update = 0
        cursor = {"next_update": 1}
        best_score = None
    else:
        assert checkpoint_dir is not None
        preview = torch.load(checkpoint_dir / "training_state.pt", map_location="cpu", weights_only=False)
        run_id = preview["run_id"]
        run_dir = Path(config.run_dir) / run_id
        state = load_training_state(checkpoint_dir, config=config, optimizer=optimizer,
                                    expected_data_hashes=data_hashes, run_dir=run_dir)
        if state["phase"] != "rl":
            raise ValueError("Cannot resume RL from a non-RL checkpoint")
        update = int(state["completed_update"])
        cursor = state["cursor"]
        best_score = state.get("best_score")

    update_limit = config.rl.updates if max_updates is None else min(config.rl.updates, max_updates)
    last_checkpoint: Path | None = resolve_checkpoint(resume) if resume else None
    started = time.perf_counter()
    try:
        while update < update_limit:
            batch_started = time.perf_counter()
            selected_tasks = random.sample(train_tasks, config.rl.task_pairs_per_update)
            episodes_by_task: list[list[dict[str, Any]]] = []
            policy = TournamentPolicy(model, config.choices, sample=True)
            policy_version = update
            for task_index, task in enumerate(selected_tasks):
                task_episodes = []
                for episode_index in range(config.rl.episodes_per_pair):
                    episode_id = f"update-{update + 1:06d}-task-{task_index:02d}-episode-{episode_index:02d}"
                    episode = collect_episode(
                        graph, task, policy, config, episode_id=episode_id,
                        policy_version=policy_version,
                        on_transition=lambda transition: append_jsonl(
                            run_dir / "transitions.jsonl", transition),
                    )
                    append_jsonl(run_dir / "episodes.jsonl", {
                        key: value for key, value in episode.items() if key != "transitions"
                    })
                    task_episodes.append(episode)
                episodes_by_task.append(task_episodes)

            all_episodes = [episode for group in episodes_by_task for episode in group]
            episode_count = len(all_episodes)
            optimizer.zero_grad(set_to_none=True)
            policy_loss_value = 0.0
            entropy_value = 0.0
            for task_episodes in episodes_by_task:
                returns = [float(episode["return"]) for episode in task_episodes]
                for episode_index, episode in enumerate(task_episodes):
                    baseline = (sum(returns) - returns[episode_index]) / (len(returns) - 1)
                    click_returns = reward_to_go(
                        [float(transition["reward"]) for transition in episode["transitions"]],
                        config.rl.gamma,
                    )
                    for transition, return_to_go in zip(episode["transitions"], click_returns):
                        advantage = return_to_go - baseline
                        for node in transition["nodes"]:
                            log_prob, entropy = replay_node(model, node, config)
                            if not torch.isfinite(log_prob.detach()) or not torch.isfinite(entropy.detach()):
                                raise FloatingPointError("non-finite replay log probability or entropy")
                            term = -(advantage * log_prob) / episode_count
                            if config.rl.entropy_coefficient:
                                term -= config.rl.entropy_coefficient * entropy / episode_count
                            term.backward()
                            policy_loss_value += float((-(advantage * log_prob.detach()) / episode_count).item())
                            entropy_value += float(entropy.detach().item())

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.rl.max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite RL gradient at update {update + 1}")
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1
            cursor = {"next_update": update + 1}
            batch_stats = _episode_metrics(all_episodes)
            metrics_record = {
                "schema_version": 1, "phase": "rl", "update": update,
                "policy_loss": policy_loss_value,
                "mean_entropy": entropy_value / max(1, batch_stats["node_count"]),
                "gradient_norm": float(grad_norm.detach().item()),
                "elapsed_seconds": time.perf_counter() - batch_started,
                **batch_stats,
            }
            append_jsonl(run_dir / "metrics.jsonl", metrics_record)
            print(
                f"RL update {update}: mean_return={batch_stats['mean_return']:.4f} "
                f"success_rate={batch_stats['success_rate']:.3f}",
                flush=True,
            )

            improved = False
            if update % config.save.evaluate_every_updates == 0 or update == update_limit:
                dev_metrics = _evaluate(model, graph, dev_tasks, config)
                append_jsonl(run_dir / "metrics.jsonl", {
                    "schema_version": 1, "phase": "rl", "event": "evaluation",
                    "update": update,
                    **{key: value for key, value in dev_metrics.items() if key != "episodes"},
                })
                score = {"success_rate": dev_metrics["success_rate"],
                         "mean_success_steps": dev_metrics["mean_success_steps"]}
                improved = best_score is None or score["success_rate"] > best_score["success_rate"] or (
                    score["success_rate"] == best_score["success_rate"] and
                    (score["mean_success_steps"] if score["mean_success_steps"] is not None else float("inf")) <
                    (best_score["mean_success_steps"] if best_score["mean_success_steps"] is not None else float("inf")))
                if improved:
                    best_score = score
                write_json_atomic(run_dir / "metrics.json", {
                    key: value for key, value in dev_metrics.items() if key != "episodes"
                })

            if (update % config.save.checkpoint_every_updates == 0 or
                    update == update_limit or improved):
                last_checkpoint = save_training_checkpoint(
                    model=model, optimizer=optimizer, config=config, run_dir=run_dir,
                    run_id=run_id, phase="rl", update=update, cursor=cursor,
                    best_score=best_score, data_hashes=data_hashes,
                    make_best=improved,
                )
                print(f"RL checkpoint: {last_checkpoint}", flush=True)

        if last_checkpoint is None:
            raise ValueError("RL run completed without an optimizer update or resumable checkpoint")
        return last_checkpoint
    except BaseException as exc:
        write_json_atomic(run_dir / "error.json", {
            "schema_version": 1, "phase": "rl", "error_type": type(exc).__name__,
            "message": str(exc), "update": update,
            "last_checkpoint": str(last_checkpoint) if last_checkpoint else None,
        })
        raise
