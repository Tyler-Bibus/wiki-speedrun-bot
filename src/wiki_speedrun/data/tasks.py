"""Reachability helpers and persisted start/target tasks."""

from __future__ import annotations

import json
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

from .load import WikiGraph


@dataclass
class Task:
    start: str
    target: str


def shortest_path(graph: WikiGraph, start: str, target: str) -> list[str] | None:
    if start not in graph.articles or target not in graph.articles:
        return None
    queue = deque([start])
    previous: dict[str, str | None] = {start: None}
    while queue:
        node = queue.popleft()
        if node == target:
            path = []
            while node is not None:
                path.append(node)
                node = previous[node]  # type: ignore[assignment]
            return list(reversed(path))
        for neighbor in graph.links.get(node, []):
            if neighbor not in previous:
                previous[neighbor] = node
                queue.append(neighbor)
    return None


def make_tasks(graph: WikiGraph, count: int = 10, min_distance: int = 2,
               max_distance: int = 3, seed: int = 42,
               max_options: int = 255) -> list[Task]:
    """Sample pairs by full-graph distance with a bounded-degree route.

    Pair distance is the shortest distance in the full directed graph. A pair is
    eligible only if some route of at most ``max_distance`` steps also exists
    using intermediate pages whose outgoing link count is at most max_options.
    Starts are shuffled and at most one task is made per start. The function
    may return fewer than ``count`` tasks when fewer starts have eligible pairs.
    Memory use is bounded to one start's BFS state and its eligible targets.
    """
    if (count <= 0 or min_distance < 1 or max_distance < min_distance
            or max_options < 0):
        return []
    rng = random.Random(seed)
    starts = sorted(graph.articles)
    rng.shuffle(starts)
    selected: list[Task] = []
    for start in starts:
        # Measure the requested distance without degree restrictions.
        distances = {start: 0}
        queue = deque([start])
        while queue:
            node = queue.popleft()
            distance = distances[node]
            if distance == max_distance:
                continue
            for neighbor in graph.links.get(node, []):
                if neighbor not in distances:
                    distances[neighbor] = distance + 1
                    queue.append(neighbor)

        # Discover targets reachable by a bounded-choice route.
        route_distance = {start: 0}
        route_queue = deque([start])
        eligible: set[str] = set()
        while route_queue:
            node = route_queue.popleft()
            depth = route_distance[node]
            if depth == max_distance or len(graph.links.get(node, [])) > max_options:
                continue
            for neighbor in graph.links.get(node, []):
                if neighbor in route_distance:
                    continue
                next_depth = depth + 1
                route_distance[neighbor] = next_depth
                route_queue.append(neighbor)
                true_distance = distances.get(neighbor)
                if (neighbor != start and true_distance is not None
                        and min_distance <= true_distance <= max_distance):
                    eligible.add(neighbor)
        if eligible:
            selected.append(Task(start, rng.choice(sorted(eligible))))
        if len(selected) >= count:
            return selected[:count]
    return selected


def save_tasks(tasks: list[Task], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(task) for task in tasks], ensure_ascii=False, indent=2), encoding="utf-8")


def load_tasks(path: Path) -> list[Task]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [Task(start=str(item["start"]), target=str(item["target"])) for item in payload]
