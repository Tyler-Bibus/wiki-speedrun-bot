"""Small, transparent reward and return calculations for navigation."""

from __future__ import annotations

from typing import Any


def click_reward(termination_reason: str | None, config: Any) -> float:
    """Return the reward for one actual navigation click."""
    reward = -float(config.reward.step_cost)
    if termination_reason == "success":
        reward += float(config.reward.success)
    elif termination_reason in {"step_limit", "dead_end"}:
        reward += float(config.reward.failure)
    elif termination_reason is not None:
        raise ValueError(f"Unknown termination reason: {termination_reason}")
    return reward


def episode_return(episode: dict[str, Any], config: Any) -> float:
    """Calculate the plan's return from an episode record.

    A zero-click dead end is still a failure, while success always takes
    precedence over a step limit because the environment checks the target first.
    """
    steps = int(episode.get("steps", len(episode.get("transitions", []))))
    reason = episode.get("termination_reason")
    if reason == "success":
        return float(config.reward.success) - float(config.reward.step_cost) * steps
    if reason in {"step_limit", "dead_end"}:
        return float(config.reward.failure) - float(config.reward.step_cost) * steps
    raise ValueError(f"Episode is not terminal: {reason!r}")


def reward_to_go(rewards: list[float], gamma: float = 1.0) -> list[float]:
    """Return the undiscounted (or explicitly discounted) suffix sums."""
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be between zero and one")
    result = [0.0] * len(rewards)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        result[index] = running
    return result
