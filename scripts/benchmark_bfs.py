#!/usr/bin/env python3
"""Benchmark full-graph BFS or move-capped BFS/DFS without changing remote resumes."""

import argparse
from collections import deque
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time
from types import SimpleNamespace

from wiki_speedrun.data.load import load_graph
from wiki_speedrun.data.tasks import shortest_path
from wiki_speedrun.remote import selected_tasks
from wiki_speedrun.remote_config import load_remote_config
from wiki_speedrun.training.checkpoints import graph_fingerprint, object_fingerprint, sha256_file


def navigate_search(graph, task, max_steps, algorithm):
    """Explore online; every forward or return edge consumes one move.

    Only visited pages expose adjacency. BFS/DFS order the discovered frontier;
    reaching that frontier requires legal travel through the revealed graph.
    Unreachable frontier entries remain pending in case later links reach them.
    """
    if algorithm not in ('bfs', 'dfs'):
        raise ValueError('Navigation algorithm must be bfs or dfs')
    if max_steps < 1:
        raise ValueError('max_steps must be positive')
    if task.start not in graph.articles or task.target not in graph.articles:
        raise ValueError('Task article absent from graph')
    path = [task.start]
    actions = []
    observed = {}
    discovered = {task.start}
    frontier = []
    backtrack_moves = 0
    reason = 'step_limit'
    while True:
        current = path[-1]
        if current == task.target:
            reason = 'success'
            break
        if len(actions) >= max_steps:
            break
        if current not in observed:
            # This is the only access to the real graph's outgoing links.
            observed[current] = list(graph.links.get(current, []))
            new = []
            for neighbor in observed[current]:
                if neighbor not in discovered:
                    discovered.add(neighbor)
                    new.append(neighbor)
            frontier.extend(new if algorithm == 'bfs' else reversed(new))
        if task.target in observed[current]:
            next_page = task.target
        else:
            # Plan travel using revealed edges only. Unvisited pages cannot be
            # expanded here, so they can only be endpoints of a known route.
            previous = {current: None}
            queue = deque([current])
            while queue:
                node = queue.popleft()
                for neighbor in observed.get(node, []):
                    if neighbor not in previous:
                        previous[neighbor] = node
                        if neighbor in observed:
                            queue.append(neighbor)
            frontier = [node for node in frontier if node not in observed]
            candidates = frontier if algorithm == 'bfs' else reversed(frontier)
            destination = next((node for node in candidates if node in previous), None)
            if destination is None:
                reason = 'dead_end' if not observed[current] else 'frontier_unreachable'
                break
            next_page = destination
            while previous[next_page] != current:
                next_page = previous[next_page]
        returning = next_page in observed
        backtrack_moves += returning
        actions.append({'source': current, 'target': next_page,
                        'kind': 'revisit' if returning else 'explore'})
        path.append(next_page)
    return {
        **asdict(task), 'path': path, 'actions': actions, 'steps': len(actions),
        'success': path[-1] == task.target, 'termination_reason': reason,
        'backtrack_moves': backtrack_moves, 'expanded_pages': len(observed),
    }


def time_navigation(graph, tasks, max_steps, algorithm):
    tasks = list(tasks)
    if not tasks:
        raise ValueError('The task list must be nonempty')
    episodes = []
    started = time.perf_counter_ns()
    for task in tasks:
        query_started = time.perf_counter_ns()
        episode = navigate_search(graph, task, max_steps, algorithm)
        episode['wall_seconds'] = (time.perf_counter_ns() - query_started) / 1e9
        episodes.append(episode)
    wall_seconds = (time.perf_counter_ns() - started) / 1e9
    latencies = sorted(e['wall_seconds'] for e in episodes)
    successful = [e for e in episodes if e['success']]
    return {
        'episode_count': len(tasks), 'success_count': len(successful),
        'success_rate': len(successful) / len(tasks), 'max_steps': max_steps,
        'wall_seconds': wall_seconds,
        'mean_wall_seconds_per_task': wall_seconds / len(tasks),
        'total_query_wall_seconds': sum(latencies),
        'median_query_wall_seconds': statistics.median(latencies),
        'p95_query_wall_seconds': latencies[math.ceil(.95 * len(latencies)) - 1],
        'mean_steps': sum(e['steps'] for e in episodes) / len(tasks),
        'mean_success_steps': (sum(e['steps'] for e in successful) / len(successful)
                               if successful else None),
        'total_backtrack_moves': sum(e['backtrack_moves'] for e in episodes),
        'failure_counts': {reason: sum(e['termination_reason'] == reason for e in episodes)
                           for reason in ('step_limit', 'dead_end', 'frontier_unreachable')},
        'episodes': episodes,
    }


def time_bfs(graph, tasks, max_steps):
    """Run a fresh directed BFS for every pair, including unsuccessful pairs."""
    tasks = list(tasks)
    if not tasks:
        raise ValueError('The task list must be nonempty')
    if max_steps < 1:
        raise ValueError('max_steps must be positive')
    for task in tasks:
        if task.start not in graph.articles or task.target not in graph.articles:
            raise ValueError('Task article absent from graph')

    episodes = []
    started = time.perf_counter_ns()
    for task in tasks:
        query_started = time.perf_counter_ns()
        route = shortest_path(graph, task.start, task.target)
        wall_seconds = (time.perf_counter_ns() - query_started) / 1e9
        steps = len(route) - 1 if route is not None else None
        episodes.append({
            **asdict(task), 'path': route, 'shortest_path_steps': steps,
            'success': steps is not None and steps <= max_steps,
            'wall_seconds': wall_seconds,
        })
    wall_seconds = (time.perf_counter_ns() - started) / 1e9
    latencies = sorted(e['wall_seconds'] for e in episodes)
    successes = sum(e['success'] for e in episodes)
    return {
        'episode_count': len(tasks), 'success_count': successes,
        'success_rate': successes / len(tasks), 'max_steps': max_steps,
        'unreachable_count': sum(e['path'] is None for e in episodes),
        'wall_seconds': wall_seconds,
        'mean_wall_seconds_per_task': wall_seconds / len(tasks),
        'total_query_wall_seconds': sum(latencies),
        'median_query_wall_seconds': statistics.median(latencies),
        'p95_query_wall_seconds': latencies[math.ceil(.95 * len(latencies)) - 1],
        'episodes': episodes,
    }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--algorithm', choices=['full-bfs', 'bfs', 'dfs'], default='full-bfs',
                        help='Full-graph planner (default), or online move-capped navigation')
    result.add_argument('--max-steps', type=int,
                        help='Move budget (default: selected config max_steps, normally 12)')
    config = result.add_mutually_exclusive_group()
    config.add_argument('--config', type=Path,
                        help='Remote config (default: configs/jev.toml)')
    config.add_argument('--training-config', type=Path,
                        help='Match benchmark --training-config, e.g. configs/snap-full.toml')
    selection = result.add_mutually_exclusive_group()
    selection.add_argument('--splits', nargs='+', choices=['dev', 'test', 'train'],
                           help='Frozen splits (default: dev test)')
    selection.add_argument('--tasks', type=Path, help='Explicit custom task file')
    result.add_argument('--max-tasks', type=int, help='Same ordered prefix as model evaluation')
    result.add_argument('--output', type=Path,
                        help='Save JSON to a new file; existing files are never overwritten')
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.max_tasks is not None and args.max_tasks < 1:
            raise ValueError('--max-tasks must be positive')
        if args.max_steps is not None and args.max_steps < 1:
            raise ValueError('--max-steps must be positive')
        splits = args.splits or (['custom'] if args.tasks else ['dev', 'test'])
        if len(set(splits)) != len(splits):
            raise ValueError('Splits must be unique')
        if args.output and args.output.exists():
            raise ValueError('Output already exists; choose a new file')
        config_path = args.training_config or args.config or Path('configs/jev.toml')
        if args.training_config:
            from wiki_speedrun.training.config import load_training_config
            training = load_training_config(config_path)
            config = SimpleNamespace(
                data_dir=training.data_dir, max_steps=training.rl.max_steps,
                dev_tasks=training.data.dev_tasks, test_tasks=training.data.test_tasks,
                train_tasks=training.data.rl_train_tasks,
                split_manifest=training.data.split_manifest,
            )
        else:
            config = load_remote_config(config_path)
        max_steps = args.max_steps if args.max_steps is not None else config.max_steps
        # Use the evaluator's frozen-manifest validation and ordered subset selection.
        selections = {split: selected_tasks(config, None if args.tasks else split,
                                            args.tasks, args.max_tasks) for split in splits}
        started = time.perf_counter_ns()
        graph = load_graph(Path(config.data_dir))
        graph_load_wall_seconds = (time.perf_counter_ns() - started) / 1e9
        report = {
            'algorithm': ('independent_directed_bfs' if args.algorithm == 'full-bfs'
                          else 'online_' + args.algorithm + '_navigation'), 'schema_version': 1,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'python': platform.python_version(), 'platform': platform.platform(),
            'config': str(config_path), 'config_sha256': sha256_file(config_path),
            'script_sha256': sha256_file(Path(__file__)),
            'graph_sha256': graph_fingerprint(graph),
            'graph_load_wall_seconds': graph_load_wall_seconds,
            'timing_scope': 'Sequential queries on a loaded graph, including path reconstruction '
                            'and per-task bookkeeping; excludes loading, validation, hashing and output. '
                            'Each query starts a fresh BFS without caching or shared search state.',
            'success_scope': 'Full-graph search; success requires shortest path within max_steps. '
                             'BFS has access to the entire adjacency graph.',
            'splits': {},
        }
        if args.algorithm != 'full-bfs':
            report.update(
                timing_scope='Sequential online navigation episodes including all search, route '
                             'planning and move bookkeeping; excludes graph loading, validation, '
                             'hashing and output. Fresh search state for every task.',
                success_scope='Reach the target within max_steps actual directed-link moves. '
                              'Every revisit/backtracking edge counts. Only visited pages expose '
                              'outgoing links. No browser-back actions or frontier teleports.',
                search_order='Source link order; FIFO discovered frontier for BFS; LIFO frontier '
                             'with first-link-first expansion for DFS. Choose the first frontier '
                             'page reachable through revealed edges, travel by its shortest known '
                             'route. Always click the target when visible on the current page.',
            )
        for split, (tasks, selection) in selections.items():
            report['splits'][split] = {
                'selection': selection,
                'ordered_tasks_sha256': object_fingerprint([asdict(t) for t in tasks]),
                **(time_bfs(graph, tasks, max_steps) if args.algorithm == 'full-bfs'
                   else time_navigation(graph, tasks, max_steps, args.algorithm)),
            }
        encoded = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + '\n'
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open('x', encoding='utf-8') as stream:
                stream.write(encoded)
        print(encoded, end='')
        return 0
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as error:
        print(f'Error: {error}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
