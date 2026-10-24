"""Concurrent independent remote episodes, with durable completion and shared limits."""
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
from queue import Queue, Empty
from threading import Event, Lock, RLock, local
import time
import uuid

from .data.load import load_graph
from .evaluation import summarize
from .models.openrouter import RemoteError, finite
from .navigation import evaluate_policy
from .remote import append, make_client, make_policy, read_rows, recover_tail, source_identity, usage_metrics
from .remote_policy import PROMPT, CHOICE_SCHEMA
from .training.checkpoints import graph_fingerprint, object_fingerprint, write_json_atomic


IDENTITY_KEYS = ('protocol', 'protocol_version', 'graph_sha256', 'ordered_tasks_sha256',
                 'selection', 'tokenizer', 'requested_model', 'settings_sha256',
                 'prompt_sha256', 'schema_sha256')
TRANSIENT = {'network', 'provider_unavailable', 'rate_limit'}


def source_compatible(folder, old, current_hash):
    if old['source_sha256'] == current_hash:
        return True
    path = Path(folder) / 'execution-upgrades.json'
    if not path.exists():
        return False
    try:
        identity = object_fingerprint({key: old[key] for key in IDENTITY_KEYS})
        return any(row.get('from_source_sha256') == old['source_sha256']
                   and row.get('to_source_sha256') == current_hash
                   and row.get('identity_sha256') == identity
                   for row in json.loads(path.read_text()).get('upgrades', []))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def completed_episodes(folder, tasks):
    """Read the ordered legacy log plus durable out-of-order completions."""
    folder = Path(folder)
    prefix = read_rows(folder / 'episodes.jsonl')
    completed = {}
    def validate(index, episode):
        if not 0 <= index < len(tasks):
            raise ValueError('Completed task index outside selected tasks')
        task = tasks[index]
        if (episode['start'], episode['target']) != (task.start, task.target):
            raise ValueError('Completed task identity mismatch')
        if episode['episode_id'] != f'eval-{index:06d}':
            raise ValueError('Completed episode ID mismatch')
        if index in completed and completed[index] != episode:
            raise ValueError('Conflicting duplicate completed episode')
        completed[index] = episode
    for index, episode in enumerate(prefix):
        validate(index, episode)
    for path in sorted((folder / 'completed').glob('*.json')):
        validate(int(path.stem), json.loads(path.read_text()))
    return completed, len(prefix)


def run_remote_parallel(config, tasks, selection, *, workers, graph=None, client=None,
                        tokenizer_identity=None, resume=None, on_created=None,
                        before_attempt_guard=None, on_attempt_observer=None, worker_factory=None,
                        cancel_event=None):
    if type(workers) is not int or workers < 1:
        raise ValueError('workers per model must be positive')
    graph = graph or load_graph(Path(config.data_dir))
    if len({(t.start, t.target) for t in tasks}) != len(tasks):
        raise ValueError('Selected tasks contain duplicate pairs')
    if any(t.start not in graph.articles or t.target not in graph.articles for t in tasks):
        raise ValueError('Task article absent from graph')
    settings = asdict(config)
    signature = {k: v for k, v in settings.items()
                 if k not in ('max_requests', 'max_cost', 'credentials_file', 'run_dir')}
    current = {**source_identity(), 'protocol': config.protocol, 'protocol_version': 1,
               'graph_sha256': graph_fingerprint(graph),
               'ordered_tasks_sha256': object_fingerprint([asdict(t) for t in tasks]),
               'selection': selection, 'tokenizer': tokenizer_identity,
               'requested_model': config.model, 'settings_sha256': object_fingerprint(signature),
               'prompt_sha256': object_fingerprint(PROMPT), 'schema_sha256': object_fingerprint(CHOICE_SCHEMA),
               'retry_policy': {'retries': config.retries, 'timeout_retry': False, 'max_retry_after_seconds': 30}}
    if resume:
        folder = Path(resume)
        provenance = json.loads((folder / 'provenance.json').read_text())
        if any(provenance.get(k) != current[k] for k in IDENTITY_KEYS) or not source_compatible(folder, provenance, current['source_sha256']):
            raise ValueError('Resume config/model/code/graph/tasks/tokenizer identity mismatch')
    else:
        folder = Path(config.run_dir) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-remote')
        folder.mkdir(parents=True, exist_ok=False)
        provenance = current
        if client is not None and client.resolved_model:
            provenance['resolved_model'] = client.resolved_model
        write_json_atomic(folder / 'config.json', settings)
        write_json_atomic(folder / 'provenance.json', provenance)
    with (folder / 'run.lock').open('a+') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another evaluator already owns this run') from None
        if on_created:
            on_created(folder)
        return _execute(folder, config, tasks, selection, graph, provenance, current, workers,
                        before_attempt_guard, on_attempt_observer, worker_factory, cancel_event or Event())


def _execute(folder, config, tasks, selection, graph, provenance, current, workers,
             before_attempt_guard, on_attempt_observer, worker_factory, cancel_event):
    for name in ('requests', 'transitions', 'episodes', 'rejected-decisions'):
        recover_tail(folder / (name + '.jsonl'))
    completed, prefix_count = completed_episodes(folder, tasks)
    records = read_rows(folder / 'requests.jsonl')
    known_cost = usage_metrics(records)['known_cost']
    reserved = 0
    lock, rate_lock, stop = RLock(), Lock(), Event()
    pacing = local()
    errors, active = {}, set()
    session = uuid.uuid4().hex
    prior = folder / 'metrics.json'
    prior_wall = json.loads(prior.read_text()).get('wall_seconds', 0) if prior.exists() else 0
    started, next_submission = time.perf_counter(), 0.0
    status = 'running'
    final_error = None
    (folder / 'completed').mkdir(exist_ok=True)
    write_json_atomic(folder / 'parallel.json', {'workers': workers, 'format_version': 1})
    append(folder / 'sessions.jsonl', {'session': session, 'workers_per_model': workers,
                                      'completed_tasks_at_start': len(completed),
                                      'source_sha256': current['source_sha256'],
                                      'max_requests': config.max_requests, 'max_cost': config.max_cost})
    queue = Queue()
    for index in range(len(tasks)):
        if index not in completed:
            queue.put(index)

    def before_attempt():
        nonlocal reserved, next_submission
        # Preserve a model's configured pacing across its workers, including retries.
        with rate_lock:
            if cancel_event.is_set():
                stop.set()
            delay = max(0, next_submission - time.monotonic())
            pacing.wait_seconds = delay
            if stop.wait(delay):
                raise RemoteError('cancelled', 'Evaluation stopped before submission')
            if before_attempt_guard:
                before_attempt_guard()
            with lock:
                if len(records) + reserved >= config.max_requests or known_cost >= config.max_cost:
                    stop.set()
                    raise RemoteError('budget', 'Configured request or reported-cost limit reached')
                reserved += 1
            next_submission = time.monotonic() + config.min_request_interval

    def on_attempt(record):
        nonlocal reserved, known_cost
        with lock:
            reserved -= 1
            record['session'] = session
            record['throttle_wait_seconds'] = record.get('throttle_wait_seconds', 0) + pacing.wait_seconds
            append(folder / 'requests.jsonl', record)
            records.append(record)
            response = record.get('response') or {}
            usage = response.get('usage')
            cost = usage.get('cost') if isinstance(usage, dict) else None
            if finite(cost):
                known_cost += cost
            if on_attempt_observer:
                on_attempt_observer(record)
            model = response.get('model')
            if model and provenance.get('resolved_model') not in (None, model):
                stop.set()
                raise RemoteError('model_changed', 'Resolved model changed between workers')
            if model and not provenance.get('resolved_model'):
                provenance['resolved_model'] = model
                write_json_atomic(folder / 'provenance.json', provenance)

    def append_trace(client, filename, row):
        with lock:
            append(folder / filename, client.redact({**row, 'session': session}))

    def flush_prefix():
        nonlocal prefix_count
        while prefix_count in completed:
            append(folder / 'episodes.jsonl', completed[prefix_count])
            prefix_count += 1

    def worker():
        try:
            if worker_factory:
                client, policy, tokenizer = worker_factory()
            else:
                client = make_client(config)
                policy, tokenizer = make_policy(config, client)
            if tokenizer != current['tokenizer']:
                raise ValueError('Worker tokenizer identity differs')
            client.resolved_model = provenance.get('resolved_model')
            client.min_request_interval = 0
            client.before_attempt, client.on_attempt = before_attempt, on_attempt
        except Exception as exc:
            with lock:
                errors['setup'] = {'kind': getattr(exc, 'kind', 'runtime'), 'message': type(exc).__name__}
                stop.set()
            return
        while not stop.is_set() and not cancel_event.is_set():
            try:
                index = queue.get_nowait()
            except Empty:
                return
            with lock:
                active.add(index)
            class PositionedPolicy:
                def choose(self, env):
                    client.context = {'episode_id': f'eval-{index:06d}', 'step': env.steps}
                    return policy.choose(env)
            try:
                def commit(row):
                    row['session'] = session
                    with lock:
                        if index in completed:
                            raise RuntimeError('Duplicate completed task claim')
                        write_json_atomic(folder / 'completed' / f'{index:06d}.json', row)
                        completed[index] = row
                        flush_prefix()
                        print(f"[{config.model}] {len(completed)}/{len(tasks)} task {index}: {row['termination_reason']}", flush=True)
                evaluate_policy(PositionedPolicy(), graph, [tasks[index]], config.max_steps, start_index=index,
                                on_transition=lambda row: append_trace(client, 'transitions.jsonl', row),
                                on_rejected=lambda row: append_trace(client, 'rejected-decisions.jsonl', row),
                                on_episode=commit)
            except Exception as exc:
                kind = getattr(exc, 'kind', 'runtime')
                with lock:
                    if kind != 'cancelled':
                        errors[str(index)] = {'kind': kind, 'message': str(exc) if isinstance(exc, RemoteError) else type(exc).__name__}
                    if kind not in TRANSIENT:
                        stop.set()
            finally:
                with lock:
                    active.discard(index)
                queue.task_done()

    def save_metrics():
        with lock:
            episodes = [completed[i] for i in sorted(completed)]
            snapshot = list(records)
            failures, working = dict(errors), sorted(active)
        metrics = summarize(episodes, graph)
        usage = usage_metrics(snapshot)
        calls = sum(e['model_calls'] for e in episodes)
        clicks = sum(e['steps'] for e in episodes)
        failure_counts = {reason: sum(e['termination_reason'] == reason for e in episodes)
                          for reason in ('step_limit', 'dead_end', 'invalid_link', 'invalid_page', 'invalid_response')}
        invalid = sum(failure_counts[k] for k in ('invalid_link', 'invalid_page', 'invalid_response'))
        metrics.update({**usage, 'status': status, 'protocol': config.protocol, 'model': config.model,
                        'selection': selection, 'intended_tasks': len(tasks), 'completed_tasks': len(episodes),
                        'split_coverage': len(episodes) / selection.get('split_count', len(tasks)),
                        'subset': selection.get('subset', False), 'wall_seconds': prior_wall + time.perf_counter() - started,
                        'completed_episode_model_calls': calls, 'total_legal_clicks': clicks,
                        'submitted_model_calls': sum(r['attempt'] == 1 for r in snapshot),
                        'invalid_decision_rate': invalid / calls if calls else None,
                        'failure_counts': failure_counts,
                        'invalid_episode_rate': invalid / len(episodes) if episodes else None,
                        'mean_remote_call_latency_ms': usage['remote_latency_ms'] / len(snapshot) if snapshot else None,
                        'mean_click_latency_ms': metrics['total_inference_latency_ms'] / clicks if clicks else None,
                        'known_cost_per_episode': usage['known_cost'] / len(episodes) if episodes else None,
                        'known_cost_per_solved_task': usage['known_cost'] / metrics['success_count'] if metrics['success_count'] else None,
                        'workers_per_model': workers, 'failed_tasks': failures, 'run_dir': str(folder)})
        write_json_atomic(folder / 'metrics.json', metrics)
        write_json_atomic(folder / 'status.json', {'status': status, 'error': final_error,
                                                 'completed_tasks': len(episodes), 'intended_tasks': len(tasks),
                                                 'active_task_indices': working, 'failed_tasks': failures})
        return metrics

    with lock:
        flush_prefix()
    save_metrics()
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(worker) for _ in range(min(workers, queue.qsize()))}
    try:
        while futures:
            done, futures = wait(futures, timeout=5, return_when=FIRST_COMPLETED)
            if cancel_event.is_set():
                stop.set()
                final_error = {'kind': 'interrupted', 'message': 'Stopped by user'}
            for future in done:
                future.result()
            save_metrics()
    except (Exception, KeyboardInterrupt) as exc:
        stop.set()
        final_error = {'kind': 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'runtime',
                       'message': type(exc).__name__}
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    with lock:
        flush_prefix()
    if len(completed) == len(tasks):
        status = 'complete'
    else:
        status = 'incomplete'
        fatal = [error for error in errors.values() if error['kind'] not in TRANSIENT]
        final_error = final_error or next(iter(fatal or list(errors.values())),
                                          {'kind': 'interrupted', 'message': 'Unfinished tasks remain'})
    return save_metrics()
