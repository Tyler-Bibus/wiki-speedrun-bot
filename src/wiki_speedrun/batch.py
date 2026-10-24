"""Durable full-split execution; resume temporary service failures within budget.

Run ``python -m wiki_speedrun.batch --background`` to detach the worker. The
ordinary benchmark command remains the foreground, single-pass reproduction.
"""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from .training.checkpoints import write_json_atomic


RETRYABLE = {'network', 'rate_limit', 'daily_quota', 'provider_unavailable'}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def retry_delay(errors):
    """Retry infrastructure interruptions, preserving benchmark episode semantics."""
    primary = [v for v in errors.values() if v.get('kind') != 'setup_dependency']
    if not primary:
        return None
    for error in primary:
        if error.get('kind') in RETRYABLE:
            continue
        if error.get('kind') == 'http' and any(f'HTTP {n}' in error.get('message', '') for n in (500, 502, 503, 504, 529)):
            continue
        return None
    return 3600 if any(e.get('kind') == 'daily_quota' for e in primary) else 300


def command(args):
    result = [sys.executable, '-m', 'wiki_speedrun', 'benchmark', '--output', str(args.output),
              '--max-total-cost', str(args.max_total_cost), '--models', *args.models,
              '--splits', *args.splits]
    if not args.include_local:
        result.append('--no-include-local')
    result += evaluation_options(args)
    return result


def evaluation_options(args):
    result = ['--remote-workers', str(getattr(args, 'remote_workers', 3)),
              '--workers-per-model', str(getattr(args, 'workers_per_model', 1))]
    for flag in ('training_config', 'sft_checkpoint'):
        value = getattr(args, flag, None)
        if value is not None:
            result += ['--' + flag.replace('_', '-'), str(value)]
    return result


def full_coverage(state, models, splits, counts, include_local):
    names = list(models) + (['kev', 'sft', 'rl'] if include_local else [])
    if state.get('settings', {}).get('max_tasks') is not None:
        return False
    for name in names:
        for split in splits:
            folder = state.get('runs', {}).get(name + '-' + split)
            if not folder:
                return False
            metrics = json.loads((Path(folder) / 'metrics.json').read_text())
            if metrics.get('status') != 'complete' or metrics.get('episode_count') != counts[split]:
                return False
    return True


def publish_report(args, job):
    """Write a full report only after the child validated every intended episode."""
    budget = json.loads((args.output / 'budget.json').read_text())
    text = ['# Full validation and test comparison', '',
            'Completed at ' + utc_now() + '.', '',
            'Every requested system completed every pair in each declared frozen split. '
            'No task sampling was used. The original five-task pilot is superseded by these results.', '',
            'Reproduce or resume the same full experiment:', '', '```bash',
            shlex.join(command(args)), '```', '',
            'MiniMax uses the explicitly labeled `llm_full_list_json_object` protocol with strict local validation. '
            'DeepSeek uses strict JSON schema. Decision systems use matched Kev tournaments. '
            'These protocols are compared as systems on identical graph/task data.', '',
            'Known suite cost (including probes and abandoned attempts): '
            f"${budget['known_cost']:.6f}. Unknown-cost attempts: {budget['unknown_cost_attempts']}. "
            f"Reported-cost limit: ${budget['max_total_cost']:.2f}.", '']
    for split in args.splits:
        content = (args.output / (split + '-results.md')).read_text()
        text += [content.replace('# Navigation comparison', '## ' + ('Validation' if split == 'dev' else 'Test')), '']
    text += ['The pilot implementation report is retained at `docs/jev-results.md`. '
             'Raw per-attempt traces, checkpoint/data/source hashes, and paired statistics are in '
             f'`{args.output}`. The SFT test outcomes had been viewed before this experiment; '
             'test outcomes did not select checkpoints or prompts.', '']
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text('\n'.join(text))
    job['report'] = str(args.report)


def worker(args):
    args.output.mkdir(parents=True, exist_ok=True)
    # The kernel releases this lock on crash; an obsolete PID file cannot block resume.
    with (args.output / 'worker.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('A full-benchmark worker already owns this output directory.', flush=True)
            return 1
        task_paths = {split: Path(f'data/splits/training/{split}_tasks.json') for split in args.splits}
        if getattr(args, 'training_config', None):
            from .training.config import load_training_config
            config = load_training_config(args.training_config)
            task_paths = {split: Path(getattr(config.data, split + '_tasks')) for split in args.splits}
        counts = {split: len(json.loads(task_paths[split].read_text()))
                  for split in args.splits}
        child = command(args)
        job = {'pid': os.getpid(), 'status': 'running', 'started_at': utc_now(),
               'updated_at': utc_now(), 'command': child, 'models': args.models,
               'split_task_counts': counts, 'max_total_cost': args.max_total_cost,
               'attempt': 0, 'report': str(args.report)}
        def save():
            job['updated_at'] = utc_now()
            write_json_atomic(args.output / 'job.json', job)
        save()
        while True:
            job.update(status='running', attempt=job['attempt'] + 1, next_retry_at=None)
            save()
            print(f"Full benchmark pass {job['attempt']} started at {utc_now()}", flush=True)
            completed = subprocess.run(child)
            job['last_exit_code'] = completed.returncode
            state_path = args.output / 'benchmark.json'
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            job['errors'] = state.get('errors', {})
            if completed.returncode == 0 and full_coverage(state, args.models, args.splits, counts, args.include_local):
                publish_report(args, job)
                job['status'] = 'complete'
                save()
                print('Full frozen-split benchmark and report complete.', flush=True)
                return 0
            delay = retry_delay(job['errors'])
            if delay is None:
                job['status'] = 'needs_attention'
                save()
                print('Full benchmark stopped on a budget or nontransient error; saved work is retained.', flush=True)
                return 1
            deadline = time.time() + delay
            job.update(status='waiting_for_service', next_retry_at=datetime.fromtimestamp(deadline, timezone.utc).isoformat())
            save()
            print(f'Temporary service limit; resuming from completed episodes in {delay} seconds.', flush=True)
            while time.time() < deadline:
                time.sleep(min(60, max(0, deadline - time.time())))
                save()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/openrouter-paid-full'))
    parser.add_argument('--report', type=Path, default=Path('docs/jev-paid-full-results.md'))
    parser.add_argument('--max-total-cost', type=float, default=9.0)
    parser.add_argument('--models', nargs='+', default=['minimax', 'deepseek', 'jev'],
                        choices=['minimax', 'deepseek', 'jev'])
    parser.add_argument('--splits', nargs='+', default=['dev', 'test'], choices=['dev', 'test'])
    parser.add_argument('--include-local', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--remote-workers', type=int, default=3)
    parser.add_argument('--workers-per-model', type=int, default=1)
    parser.add_argument('--training-config', type=Path)
    parser.add_argument('--sft-checkpoint', type=Path)
    parser.add_argument('--background', action='store_true')
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_total_cost) or args.max_total_cost <= 0:
        parser.error("--max-total-cost must be finite and positive")
    if args.remote_workers < 1:
        parser.error('--remote-workers must be positive')
    if args.workers_per_model < 1:
        parser.error('--workers-per-model must be positive')
    if args.background:
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / 'worker.lock').open('a+') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f'Full benchmark already running; status {args.output / "job.json"}')
                return 0
        child = [sys.executable, '-m', 'wiki_speedrun.batch', '--output', str(args.output),
                 '--report', str(args.report), '--max-total-cost', str(args.max_total_cost),
                 '--models', *args.models, '--splits', *args.splits]
        if not args.include_local:
            child.append('--no-include-local')
        child += evaluation_options(args)
        with (args.output / 'worker.log').open('ab', buffering=0) as log:
            process = subprocess.Popen(child, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        print(f'Full benchmark launched: PID {process.pid}; status {args.output / "job.json"}; log {args.output / "worker.log"}')
        return 0
    return worker(args)


if __name__ == '__main__':
    raise SystemExit(main())
