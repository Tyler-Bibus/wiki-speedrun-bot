"""Optional in-process training for the tournament navigation policy.

This package deliberately avoids importing PyTorch at package import time so
the standard-library HTTP baseline remains usable without the training extra.
"""

__all__ = ["collect_episode", "episode_return", "train_rl", "train_sft", "evaluate_trained"]


def __getattr__(name):
    if name == "collect_episode":
        from .rollouts import collect_episode
        return collect_episode
    if name in {"train_rl", "evaluate_trained"}:
        from .rl import evaluate_trained, train_rl
        return {"train_rl": train_rl, "evaluate_trained": evaluate_trained}[name]
    if name == "train_sft":
        from .sft import train_sft
        return train_sft
    if name == "episode_return":
        from .rewards import episode_return
        return episode_return
    raise AttributeError(name)
