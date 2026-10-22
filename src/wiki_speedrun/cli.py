"""Command-line entry points for preparing data and running the baseline."""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

from .config import load_config
from .data.download import prepare_data
from .data.load import load_graph
from .data.tasks import Task, load_tasks, make_tasks, save_tasks
from .evaluation import summarize
from .models.kev import KevClient
from .policy import KevPolicy, RandomPolicy
from .runner import run_episode


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description="Run the untuned Kev Wikispeedia baseline")
    result.add_argument("--config", type=Path, default=Path("configs/baseline.toml"))
    commands = result.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-data", help="Download and prepare SNAP data")
    prepare.add_argument("--no-text", action="store_true", help="Download only the graph and paths")
    commands.add_parser("smoke-test", help="Ask the running Kev server one simple question")
    tasks = commands.add_parser("make-tasks", help="Save fixed reachable evaluation pairs")
    tasks.add_argument("--count", type=int, default=10)
    tasks.add_argument("--min-distance", type=int, default=2)
    tasks.add_argument("--max-distance", type=int, default=3)
    tasks.add_argument("--output", type=Path, default=Path("data/splits/baseline.json"))
    play = commands.add_parser("play", help="Navigate from a given article to a target")
    play.add_argument("start", help="Article title, with spaces or underscores")
    play.add_argument("target")
    play.add_argument("--policy", choices=["kev", "random"], default="kev")
    evaluate = commands.add_parser("evaluate", help="Run a fixed set of tasks")
    evaluate.add_argument("--tasks", type=Path, default=Path("data/splits/baseline.json"))
    evaluate.add_argument("--policy", choices=["kev", "random"], default="kev")
    return result


def save_run(config, episodes, metrics, policy_name):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    folder = Path(config.run_dir) / f"{stamp}-{policy_name}"
    folder.mkdir(parents=True)
    with (folder / "episodes.jsonl").open("w", encoding="utf-8") as handle:
        for episode in episodes:
            handle.write(json.dumps(episode, ensure_ascii=False) + "\n")
    for name, value in [("metrics", metrics), ("config", {**asdict(config), "policy": policy_name})]:
        (folder / f"{name}.json").write_text(json.dumps(value, indent=2), encoding="utf-8")
    return folder


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "prepare-data":
            print(prepare_data(Path(config.data_dir), include_text=not args.no_text))
            return 0
        if args.command == "smoke-test":
            answer = KevClient(config.base_url, config.timeout).smoke_test()
            print(json.dumps(answer, indent=2))
            if answer["choice"] != "earth":
                print("Kev responded, but did not choose the expected answer.", file=sys.stderr)
                return 1
            return 0

        graph = load_graph(Path(config.data_dir))
        if args.command == "make-tasks":
            tasks = make_tasks(graph, args.count, args.min_distance, args.max_distance,
                               config.seed, config.max_options)
            if len(tasks) < args.count:
                raise ValueError(f"Only found {len(tasks)} suitable tasks; requested {args.count}")
            save_tasks(tasks, args.output)
            print(f"Saved {len(tasks)} tasks to {args.output}")
            return 0

        if args.command == "play":
            tasks = [Task(args.start.replace(" ", "_"), args.target.replace(" ", "_"))]
        else:
            tasks = load_tasks(args.tasks)
        if not tasks:
            raise ValueError("The task file contains no tasks")
        if args.policy == "kev":
            policy = KevPolicy(KevClient(config.base_url, config.timeout),
                               config.excerpt_chars, config.max_options)
        else:
            policy = RandomPolicy(config.seed)
        episodes = []
        for task in tasks:
            episode = run_episode(graph, task, policy, config.max_steps)
            episodes.append(episode)
            print(f"{task.start} -> {task.target}: {episode['termination_reason']} "
                  f"({episode['steps']} clicks)", flush=True)
            print("  " + " -> ".join(episode["path"]), flush=True)
        metrics = summarize(episodes, graph)
        folder = save_run(config, episodes, metrics, args.policy)
        print(json.dumps(metrics, indent=2))
        print(f"Saved run to {folder}")
        return 0
    except (OSError, ValueError, RuntimeError, TypeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
