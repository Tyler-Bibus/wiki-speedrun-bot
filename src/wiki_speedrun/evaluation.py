"""Aggregate episode outcomes and compare successful routes to graph shortest paths."""

from .data.tasks import shortest_path


def summarize(episodes, graph):
    episodes = list(episodes)
    count = len(episodes)
    successful = [episode for episode in episodes if episode["success"]]
    steps = [episode["steps"] for episode in episodes]
    latencies = [
        decision.get("latency_ms", 0.0)
        for episode in episodes
        for decision in episode.get("decisions", [])
    ]
    revisits = [
        sum(max(0, episode["path"].count(article) - 1) for article in set(episode["path"]))
        for episode in episodes
    ]
    shortest_lengths = []
    overheads = []
    for episode in successful:
        route = shortest_path(graph, episode["start"], episode["target"])
        if route is not None:
            shortest = max(0, len(route) - 1)
            shortest_lengths.append(shortest)
            overheads.append(episode["steps"] - shortest)
    return {
        "episode_count": count,
        "success_count": len(successful),
        "success_rate": len(successful) / count if count else 0.0,
        "mean_steps": sum(steps) / count if count else 0.0,
        "total_inference_latency_ms": sum(latencies),
        "mean_inference_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
        "total_revisit_count": sum(revisits),
        "mean_revisit_count": sum(revisits) / count if count else 0.0,
        "mean_success_steps": sum(e["steps"] for e in successful) / len(successful) if successful else 0.0,
        "mean_shortest_path_steps": sum(shortest_lengths) / len(shortest_lengths) if shortest_lengths else None,
        "mean_shortest_path_overhead": sum(overheads) / len(overheads) if overheads else None,
    }
