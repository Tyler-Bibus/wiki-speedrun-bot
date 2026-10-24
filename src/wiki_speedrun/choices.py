"""Token-safe tournaments over every outgoing link."""

from __future__ import annotations

import random
import time
from typing import Any

from .data.load import display_title


INSTRUCTIONS = (
    "Reach the target using as few clicks as possible. Use the page excerpt and "
    "history to avoid loops. Choose only an outgoing link and return its link ID."
)


def _value(config: Any, name: str, default: Any) -> Any:
    return getattr(config, name, default)


def make_request(state: dict[str, Any], candidate_ids: list[str],
                 articles: list[str]) -> dict[str, Any]:
    """Build the exact ordered request for one tournament node."""
    if len(candidate_ids) != len(articles) or not candidate_ids:
        raise ValueError("candidate IDs and articles must be non-empty and parallel")
    return {
        "model": "kev-latest",
        "state": state,
        "questions": {"next_link": {
            "type": "choice",
            "instructions": INSTRUCTIONS,
            "criteria": {candidate_id: display_title(article)
                         for candidate_id, article in zip(candidate_ids, articles)},
        }},
    }


def _fit_state(model: Any, state: dict[str, Any], config: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Shorten excerpt, then oldest history, until the state limit is met."""
    shortened = {"excerpt_chars_removed": 0, "history_entries_removed": 0}
    result = dict(state)
    result["history"] = list(state.get("history", []))
    limit = int(_value(config, "max_state_tokens", 1024))
    while model.state_token_count(result) > limit:
        excerpt = result.get("current_excerpt", "")
        if excerpt:
            old_length = len(excerpt)
            excerpt = excerpt[: max(0, old_length // 2)]
            result["current_excerpt"] = excerpt
            shortened["excerpt_chars_removed"] += old_length - len(excerpt)
            continue
        history = result["history"]
        if history:
            history.pop(0)
            shortened["history_entries_removed"] += 1
            continue
        raise ValueError(f"current and target titles exceed max_state_tokens={limit}")
    return result, shortened


def _groups(model: Any, state: dict[str, Any], candidate_ids: list[str],
            articles_by_id: dict[str, str], config: Any) -> list[list[str]]:
    """Make source-ordered groups, splitting oversized groups in halves."""
    maximum = int(_value(config, "node_max_choices", 64))
    if not 2 <= maximum <= 255:
        raise ValueError("node_max_choices must be between 2 and 255")
    groups: list[list[str]] = []

    def split(group: list[str]) -> None:
        request = make_request(state, group, [articles_by_id[item] for item in group])
        if len(group) <= maximum and model.fits(request):
            groups.append(group)
            return
        if len(group) == 1:
            raise ValueError(f"single candidate {group[0]} cannot fit max_row_tokens")
        if len(group) == 2:
            raise ValueError("two candidate rows cannot fit max_row_tokens; increase the budget")
        middle = len(group) // 2
        split(group[:middle])
        split(group[middle:])

    for offset in range(0, len(candidate_ids), maximum):
        split(candidate_ids[offset:offset + maximum])
    if not groups:
        raise ValueError("cannot group an empty candidate list")
    return groups


def _predict(model: Any, request: dict[str, Any], candidate_ids: list[str],
             sample: bool) -> tuple[str, float | None, float]:
    """Run one no-gradient categorical decision and retain its audit values."""
    import torch

    with torch.no_grad():
        encoded = model.encode(request)
        encoded_ids = encoded.get("choice_ids")
        if encoded_ids is not None and list(encoded_ids) != candidate_ids:
            raise ValueError("encoded choice IDs do not match tournament candidate order")
        logits = model.logits(encoded).float()
        if logits.ndim != 1 or logits.numel() != len(candidate_ids):
            raise ValueError("model logits must match the ordered candidate list")
        if not bool(torch.isfinite(logits).all()):
            raise ValueError("model returned non-finite logits")
        log_probs = torch.log_softmax(logits, dim=0)
        probabilities = log_probs.exp()
        if sample:
            selected = int(torch.multinomial(probabilities, 1).item())
        else:
            selected = int(torch.argmax(logits).item())
        entropy = float((-(probabilities * log_probs).sum()).item())
        old_log_prob = float(log_probs[selected].item()) if sample else None
    return candidate_ids[selected], old_log_prob, entropy


class TournamentPolicy:
    """Resolve every edge through deterministic groups and successive winners."""

    def __init__(self, model: Any, config: Any, sample: bool = False, *, predictor=None):
        self.model = model
        self.config = config
        self.sample = sample
        self.predictor = predictor

    def choose(self, env: Any) -> dict[str, Any]:
        moves = env.legal_moves()
        if not moves:
            raise ValueError("cannot choose from a state with no legal moves")
        if len(moves) == 1:
            return {"article": moves[0], "selected_link_id": "link_0",
                    "nodes": [], "latency_ms": 0.0, "probabilities": {}}
        candidate_ids = [f"link_{index}" for index in range(len(moves))]
        articles_by_id = dict(zip(candidate_ids, moves))
        started = time.perf_counter()
        state = {
            "current_title": display_title(env.current),
            "target_title": display_title(env.target),
            "current_excerpt": env.graph.texts.get(env.current, "")[:int(
                _value(self.config, "excerpt_chars", 1500))],
            "history": [display_title(article) for article in env.history],
        }
        state, shortening = _fit_state(self.model, state, self.config)
        nodes: list[dict[str, Any]] = []
        round_number = 0
        while len(candidate_ids) > 1:
            groups = _groups(self.model, state, candidate_ids, articles_by_id, self.config)
            winners: list[str] = []
            for node_index, group in enumerate(groups):
                if len(group) == 1:
                    winners.append(group[0])
                    continue
                request = make_request(state, group, [articles_by_id[item] for item in group])
                if self.predictor is None:
                    winner, old_log_prob, entropy = _predict(self.model, request, group, self.sample)
                else:
                    winner, old_log_prob, entropy = self.predictor(request, group, round_number, node_index)
                winners.append(winner)
                state_tokens = int(self.model.state_token_count(state))
                token_counter = getattr(self.model, "token_counts", None)
                token_counts = token_counter(request) if token_counter is not None else {
                    "state": state_tokens, "row": None,
                }
                nodes.append({
                    "round": round_number,
                    "node": node_index,
                    "request": request,
                    "candidate_ids": list(group),
                    "articles": [articles_by_id[item] for item in group],
                    "selected_link_id": winner,
                    "old_log_prob": old_log_prob,
                    "entropy": entropy,
                    "token_counts": token_counts,
                    "state_shortening": dict(shortening),
                })
            if len(winners) >= len(candidate_ids):
                raise RuntimeError("tournament round failed to reduce candidate count")
            candidate_ids = winners
            round_number += 1
        selected = candidate_ids[0]
        return {"article": articles_by_id[selected], "selected_link_id": selected,
                "nodes": nodes, "latency_ms": (time.perf_counter() - started) * 1000.0,
                "probabilities": {}}


def build_supervised_nodes(model: Any, config: Any, state: dict[str, Any],
                           candidate_ids: list[str], articles: list[str],
                           label_id: str, seed: int) -> list[dict[str, Any]]:
    """Build target-ancestry node records for one demonstrated route step."""
    if len(candidate_ids) != len(articles) or label_id not in candidate_ids:
        raise ValueError("demonstrated label must name one of the parallel candidates")
    articles_by_id = dict(zip(candidate_ids, articles))
    state, shortening = _fit_state(model, state, config)
    rng = random.Random(seed)
    remaining = list(candidate_ids)
    records: list[dict[str, Any]] = []
    tournament_trace: list[dict[str, Any]] = []
    round_number = 0
    while len(remaining) > 1:
        groups = _groups(model, state, remaining, articles_by_id, config)
        winners: list[str] = []
        group_decisions: list[dict[str, Any]] = []
        label_group: list[str] | None = None
        supervised_node: dict[str, Any] | None = None
        for node_index, group in enumerate(groups):
            if label_id in group:
                label_group = group
            if len(group) == 1:
                winners.append(group[0])
                group_decisions.append({
                    "candidate_ids": list(group), "winner_id": group[0],
                    "model_call": False,
                })
                continue
            request = make_request(state, group, [articles_by_id[item] for item in group])
            winner = label_id if label_id in group else rng.choice(group)
            winners.append(winner)
            group_decisions.append({
                "candidate_ids": list(group), "winner_id": winner,
                "request": request, "model_call": True,
            })
            if label_id in group:
                token_counter = getattr(model, "token_counts", None)
                token_counts = token_counter(request) if token_counter is not None else {
                    "state": int(model.state_token_count(state)), "row": None,
                }
                supervised_node = {
                    "round": round_number, "node": node_index, "request": request,
                    "candidate_ids": list(group),
                    "articles": [articles_by_id[item] for item in group],
                    "label": label_id,
                    "token_counts": token_counts,
                    "state_shortening": dict(shortening),
                }
        tournament_trace.append({"round": round_number, "groups": group_decisions})
        if supervised_node is not None:
            supervised_node["group_winners"] = list(winners)
            records.append(supervised_node)
        if label_group is None or label_id not in winners:
            raise RuntimeError("demonstrated label was lost from its tournament ancestry")
        if len(winners) >= len(remaining):
            raise RuntimeError("SFT tournament round failed to reduce candidate count")
        remaining = winners
        round_number += 1
    for record in records:
        record["tournament_trace"] = tournament_trace
    return records
