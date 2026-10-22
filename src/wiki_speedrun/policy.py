"""Policies for choosing the next outgoing article."""

import random
import time

from .observations import build_request


class KevPolicy:
    def __init__(self, client, excerpt_chars=1500, max_options=255):
        self.client = client
        self.excerpt_chars = excerpt_chars
        self.max_options = max_options

    def choose(self, env):
        request = build_request(env, self.excerpt_chars, self.max_options)
        started = time.perf_counter()
        result = self.client.score(request)
        latency_ms = (time.perf_counter() - started) * 1000
        choice = result["choice"]
        try:
            index = int(choice.removeprefix("link_"))
            article = env.legal_moves()[index]
        except (ValueError, IndexError, AttributeError) as exc:
            raise ValueError(f"Model returned an invalid link choice: {choice!r}") from exc
        if choice not in request["questions"]["next_link"]["criteria"]:
            raise ValueError(f"Model returned an unknown link choice: {choice!r}")
        return {
            "article": article,
            "probabilities": dict(result.get("probabilities", {})),
            "confidence": result.get("confidence"),
            "latency_ms": latency_ms,
        }


class RandomPolicy:
    """Reproducible smoke-test baseline, uniform over legal outgoing links."""

    def __init__(self, seed=42):
        self.random = random.Random(seed)

    def choose(self, env):
        moves = env.legal_moves()
        if not moves:
            raise ValueError("Cannot choose a move from a dead end")
        index = self.random.randrange(len(moves))
        probabilities = {f"link_{i}": 1.0 / len(moves) for i in range(len(moves))}
        return {
            "article": moves[index],
            "probabilities": probabilities,
            "confidence": 1.0 / len(moves),
            "latency_ms": 0.0,
        }
