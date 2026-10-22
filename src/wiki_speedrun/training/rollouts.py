"""On-policy episode collection and node-level policy replay."""

from __future__ import annotations

import math
from contextlib import nullcontext
from typing import Any

from ..environment import NavigationEnvironment
from .rewards import click_reward, episode_return


def autocast_context(config: Any):
    """Match rollout/replay precision while leaving CPU fake models untouched."""
    device = str(config.model.device)
    dtype = str(config.model.autocast_dtype)
    if not device.startswith("cuda") or dtype == "fp32":
        return nullcontext()
    import torch

    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def collect_episode(graph: Any, task: Any, policy: Any, config: Any,
                    episode_id: str = "episode-0", policy_version: int = 0,
                    on_transition: Any | None = None) -> dict[str, Any]:
    """Collect one sampled trajectory, retaining every tournament node.

    ``policy.choose`` runs its node forwards under ``torch.no_grad``. Only
    NavigationEnvironment.step calls count as actions and receive rewards.
    """
    env = NavigationEnvironment(graph, task.start, task.target, config.rl.max_steps)
    transitions: list[dict[str, Any]] = []
    if env.done:
        record = {
            "episode_id": episode_id,
            "start": task.start,
            "target": task.target,
            "path": list(env.history),
            "steps": 0,
            "success": env.success,
            "termination_reason": env.termination_reason,
            "transitions": transitions,
            "policy_version": policy_version,
        }
        record["return"] = episode_return(record, config)
        return record

    while not env.done:
        before = env.current
        history = list(env.history)
        with autocast_context(config):
            result = policy.choose(env)
        article = result["article"]
        nodes = result.get("nodes", [])
        for node in nodes:
            value = node.get("old_log_prob")
            if value is None or not math.isfinite(float(value)):
                raise ValueError("sampled tournament node is missing a finite old_log_prob")
        env.step(article)
        reason = env.termination_reason
        reward = click_reward(reason, config)
        transitions.append({
            "schema_version": 1,
            "policy_version": policy_version,
            "episode_id": episode_id,
            "pair": {"start": task.start, "target": task.target},
            "step": env.steps - 1,
            "source": before,
            "target": task.target,
            "history": history,
            "selected_link_id": result["selected_link_id"],
            "selected_article": article,
            "reward": reward,
            "next_article": env.current,
            "termination_reason": reason,
            "nodes": nodes,
            "latency_ms": float(result.get("latency_ms", 0.0)),
        })
        if on_transition is not None:
            on_transition(transitions[-1])

    record = {
        "schema_version": 1,
        "episode_id": episode_id,
        "pair": {"start": task.start, "target": task.target},
        "start": task.start,
        "target": task.target,
        "path": list(env.history),
        "steps": env.steps,
        "success": env.success,
        "termination_reason": env.termination_reason,
        "transitions": transitions,
        "policy_version": policy_version,
    }
    record["return"] = episode_return(record, config)
    return record


def replay_node(model: Any, node: dict[str, Any], config: Any):
    """Recompute one node's chosen log probability with gradients enabled."""
    import torch

    with autocast_context(config):
        encoded = model.encode(node["request"])
        logits = model.logits(encoded).float()
    candidate_ids = node["candidate_ids"]
    if logits.ndim != 1 or logits.numel() != len(candidate_ids):
        raise ValueError("replayed logits do not match the stored candidate order")
    try:
        selected_index = candidate_ids.index(node["selected_link_id"])
    except ValueError as exc:
        raise ValueError("stored selected link is absent from candidate IDs") from exc
    log_probs = torch.log_softmax(logits, dim=0)
    current_log_prob = log_probs[selected_index]
    old_log_prob = float(node["old_log_prob"])
    tolerance = float(getattr(config.rl, "replay_tolerance", 2e-4))
    if not math.isclose(float(current_log_prob.detach().item()), old_log_prob,
                        rel_tol=tolerance, abs_tol=tolerance):
        raise RuntimeError(
            "sampled and replayed log probabilities differ: "
            f"{old_log_prob:.8g} vs {float(current_log_prob.detach().item()):.8g}"
        )
    probabilities = log_probs.exp()
    entropy = -(probabilities * log_probs).sum()
    return current_log_prob, entropy
