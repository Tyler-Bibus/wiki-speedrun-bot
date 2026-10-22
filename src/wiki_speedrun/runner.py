"""Run a policy through one navigation task."""

import time

from .environment import NavigationEnvironment
from .observations import OptionLimitError


def run_episode(graph, task, policy, max_steps=12):
    env = NavigationEnvironment(graph, task.start, task.target, max_steps=max_steps)
    decisions = []
    started = time.perf_counter()
    option_limit_error = None
    while not env.done:
        moves = env.legal_moves()
        options = {f"link_{index}": article for index, article in enumerate(moves)}
        try:
            decision = policy.choose(env)
        except OptionLimitError as exc:
            option_limit_error = str(exc)
            break
        article = decision["article"]
        source = env.current
        env.step(article)
        decisions.append({
            "source": source,
            "options": options,
            "article": article,
            "probabilities": decision.get("probabilities", {}),
            "confidence": decision.get("confidence"),
            "latency_ms": decision.get("latency_ms", 0.0),
        })
    elapsed = time.perf_counter() - started
    return {
        "start": task.start,
        "target": task.target,
        "success": env.success,
        "termination_reason": "option_limit" if option_limit_error else env.termination_reason,
        "steps": env.steps,
        "path": list(env.history),
        "decisions": decisions,
        "elapsed_seconds": elapsed,
        "error_message": option_limit_error,
    }
