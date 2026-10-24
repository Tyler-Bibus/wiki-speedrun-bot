"""Incremental remote runs with immutable episode boundaries and bounded spend."""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import uuid

from .credentials import load_api_key
from .data.load import load_graph
from .data.tasks import Task
from .evaluation import summarize
from .models.openrouter import OpenRouterChatClient, OpenRouterDecisionClient, RemoteError, finite
from .navigation import evaluate_policy
from .remote_policy import LLMPolicy, RemoteDecisionPolicy, PROMPT, CHOICE_SCHEMA
from .training.checkpoints import graph_fingerprint, object_fingerprint, sha256_file, write_json_atomic


def append(path, value):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        import os
        os.fsync(stream.fileno())


def read_rows(path):
    if not Path(path).exists():
        return []
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def recover_tail(path):
    """Preserve a crash-torn final append, then truncate only that incomplete line."""
    path = Path(path)
    if not path.exists():
        return
    raw = path.read_bytes()
    if raw and not raw.endswith(b'\n'):
        boundary = raw.rfind(b'\n') + 1
        with path.with_suffix(path.suffix + '.torn').open('ab') as stream:
            stream.write(raw[boundary:])
        path.write_bytes(raw[:boundary])


def make_client(config, **kwargs):
    key = load_api_key(config.credentials_file, config.key_variable)
    cls = OpenRouterDecisionClient if config.protocol == 'matched_kev_tournament' else OpenRouterChatClient
    return cls(key, config.model, config.timeout, config.retries, min_request_interval=config.min_request_interval, **kwargs)


def make_policy(config, client):
    if config.protocol == 'matched_kev_tournament':
        from .models.kev_budget import KevContextBudgeter
        budgeter = KevContextBudgeter(config.tokenizer, config)
        return RemoteDecisionPolicy(client, budgeter, config), budgeter.metadata
    return LLMPolicy(client, config), None


def selected_tasks(config, split=None, tasks_path=None, max_tasks=None):
    if split and tasks_path:
        raise ValueError('Use either --tasks or --split')
    if max_tasks is not None and max_tasks < 1:
        raise ValueError('--max-tasks must be positive')
    split = split or ('custom' if tasks_path else 'dev')
    path = Path(tasks_path or getattr(config, split + '_tasks'))
    values = json.loads(path.read_text())
    tasks = [Task(str(v['start']), str(v['target'])) for v in values]
    if not tasks or len({(t.start, t.target) for t in tasks}) != len(tasks):
        raise ValueError('Task list must be nonempty and contain unique pairs')
    manifest_hash = None
    if split != 'custom':
        manifest = Path(config.split_manifest)
        data = json.loads(manifest.read_text())
        expected = data['output_sha256'].get(str(path))
        if expected is None:
            expected = next((v for k, v in data['output_sha256'].items() if Path(k).resolve() == path.resolve()), None)
        if expected != sha256_file(path):
            raise ValueError('Frozen split manifest task hash mismatch')
        assigned = {(v['start'], v['target']) for v in data['pair_assignments'] if v['split'] == split}
        if assigned != {(t.start, t.target) for t in tasks}:
            raise ValueError('Frozen split pair assignments mismatch')
        manifest_hash = sha256_file(manifest)
    intended = len(tasks)
    tasks = tasks[:max_tasks] if max_tasks else tasks
    return tasks, {'split': split, 'max_tasks': max_tasks, 'subset': max_tasks is not None,
                   'split_count': intended, 'tasks_file_sha256': sha256_file(path),
                   'manifest_sha256': manifest_hash}


def source_identity():
    try:
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
        dirty = bool(subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip())
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    files = sorted(Path('src/wiki_speedrun').rglob('*.py'))
    return {'git_revision': revision, 'git_dirty': dirty,
            'source_sha256': object_fingerprint({str(p): sha256_file(p) for p in files})}


def usage_metrics(records):
    responses = [r.get('response') for r in records]
    usages = [r.get('usage') if isinstance(r, dict) else None for r in responses]
    costs = [u.get('cost') if isinstance(u, dict) else None for u in usages]
    result = {'http_attempts': len(records), 'known_cost': sum(c for c in costs if finite(c)),
              'missing_usage_count': sum(u is None for u in usages),
              'unknown_cost_attempts': sum(not finite(c) for c in costs),
              'remote_latency_ms': sum(r.get('latency_ms', 0) for r in records),
              'throttle_wait_ms': sum(r.get('throttle_wait_seconds', 0) * 1000 for r in records),
              'retry_wait_ms': sum(r.get('retry_delay_seconds', 0) * 1000 for r in records),
              'tokens': {}}
    for key in ('input_tokens', 'output_tokens', 'prompt_tokens', 'completion_tokens', 'total_tokens'):
        values = [u[key] for u in usages if isinstance(u, dict) and finite(u.get(key))]
        result['tokens'][key] = sum(values) if values else None
    for detail, key in [('prompt_tokens_details', 'cached_tokens'), ('completion_tokens_details', 'reasoning_tokens')]:
        values = [u[detail][key] for u in usages if isinstance(u, dict) and isinstance(u.get(detail), dict)
                  and finite(u[detail].get(key))]
        result['tokens'][key] = sum(values) if values else None
    result['cost_is_complete'] = not result['unknown_cost_attempts']
    return result


def run_remote(config, tasks, selection, *, graph=None, client=None, policy=None,
               tokenizer_identity=None, resume=None, on_created=None,
               before_attempt_guard=None, on_attempt_observer=None):
    graph = graph or load_graph(Path(config.data_dir))
    client = client or make_client(config)
    if policy is None:
        policy, tokenizer_identity = make_policy(config, client)
    for task in tasks:
        if task.start not in graph.articles or task.target not in graph.articles:
            raise ValueError('Task article absent from graph')
    settings = asdict(config)
    # Limits may be raised deliberately on resume; the protocol and sampling cannot change.
    signature_settings = {k: v for k, v in settings.items() if k not in ('max_requests', 'max_cost', 'credentials_file', 'run_dir')}
    provenance = {**source_identity(), 'protocol': config.protocol, 'protocol_version': 1,
                  'graph_sha256': graph_fingerprint(graph),
                  'ordered_tasks_sha256': object_fingerprint([asdict(t) for t in tasks]),
                  'selection': selection, 'tokenizer': tokenizer_identity,
                  'requested_model': config.model, 'settings_sha256': object_fingerprint(signature_settings),
                  'prompt_sha256': object_fingerprint(PROMPT), 'schema_sha256': object_fingerprint(CHOICE_SCHEMA),
                  'retry_policy': {'retries': config.retries, 'timeout_retry': False, 'max_retry_after_seconds': 30}}
    identity_keys = ('source_sha256', 'protocol', 'protocol_version', 'graph_sha256', 'ordered_tasks_sha256',
                     'selection', 'tokenizer', 'requested_model', 'settings_sha256', 'prompt_sha256', 'schema_sha256')
    prior_wall = 0
    if resume:
        folder = Path(resume)
        old = json.loads((folder / 'provenance.json').read_text())
        if any(old.get(k) != provenance[k] for k in identity_keys):
            raise ValueError('Resume config/model/code/graph/tasks/tokenizer identity mismatch')
        client.resolved_model = old.get('resolved_model')
        for name in ('requests', 'transitions', 'episodes', 'rejected-decisions'):
            recover_tail(folder / (name + '.jsonl'))
        previous_metrics = folder / 'metrics.json'
        if previous_metrics.exists():
            prior_wall = json.loads(previous_metrics.read_text()).get('wall_seconds', 0)
        provenance = old
    else:
        folder = Path(config.run_dir) / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ') + '-remote')
        folder.mkdir(parents=True, exist_ok=False)
        write_json_atomic(folder / 'config.json', settings)
        write_json_atomic(folder / 'provenance.json', provenance)
    if on_created:
        on_created(folder)
    episodes = read_rows(folder / 'episodes.jsonl')
    if len(episodes) > len(tasks):
        raise ValueError('Resume contains excess completed episodes')
    for i, episode in enumerate(episodes):
        if (episode['start'], episode['target']) != (tasks[i].start, tasks[i].target):
            raise ValueError('Resume completed task order mismatch')
    records = read_rows(folder / 'requests.jsonl')
    session = uuid.uuid4().hex
    append(folder / 'sessions.jsonl', {'session': session, 'restart_episode_index': len(episodes),
                                      'note': 'Earlier uncommitted episode traces are abandoned; spend is retained.',
                                      'max_requests': config.max_requests, 'max_cost': config.max_cost})
    progress = {'task_index': len(episodes), 'last_completed_task_index': len(episodes) - 1}
    started = time.perf_counter()
    status = 'running'
    error_info = None

    def before_attempt():
        if before_attempt_guard:
            before_attempt_guard()
        usage = usage_metrics(records)
        if len(records) >= config.max_requests or usage['known_cost'] >= config.max_cost:
            raise RemoteError('budget', 'Configured request or reported-cost limit reached')

    def on_attempt(record):
        record['session'] = session
        append(folder / 'requests.jsonl', record)
        records.append(record)
        if on_attempt_observer:
            on_attempt_observer(record)
        model = record.get('response', {}).get('model')
        if model and not provenance.get('resolved_model'):
            provenance['resolved_model'] = model
            write_json_atomic(folder / 'provenance.json', provenance)

    client.before_attempt, client.on_attempt = before_attempt, on_attempt

    class PositionedPolicy:
        def choose(self, env):
            client.context = {'episode_id': f"eval-{progress['task_index']:06d}", 'step': env.steps}
            return policy.choose(env)

    def task_start(task, index):
        progress.update(task_index=index, task=asdict(task))
        write_status()

    def write_status():
        write_json_atomic(folder / 'status.json', {'status': status, **progress, 'error': error_info,
                                                 'completed_tasks': len(episodes), 'intended_tasks': len(tasks)})

    def transition(row):
        row['session'] = session
        append(folder / 'transitions.jsonl', client.redact(row))

    def rejected(row):
        row['session'] = session
        append(folder / 'rejected-decisions.jsonl', client.redact(row))

    def episode_done(row):
        row['session'] = session
        append(folder / 'episodes.jsonl', row)
        episodes.append(row)
        progress['last_completed_task_index'] = len(episodes) - 1
        print(f"{len(episodes)}/{len(tasks)} {row['start']} -> {row['target']}: {row['termination_reason']} ({row['steps']} clicks)", flush=True)
        save_metrics()
        write_status()

    def save_metrics():
        metrics = summarize(episodes, graph)
        usage = usage_metrics(records)
        successes = metrics['success_count']
        calls = sum(e['model_calls'] for e in episodes)
        clicks = sum(e['steps'] for e in episodes)
        failure_counts = {reason: sum(e['termination_reason'] == reason for e in episodes)
                          for reason in ('step_limit', 'dead_end', 'invalid_link', 'invalid_page', 'invalid_response')}
        metrics.update({**usage, 'status': status, 'protocol': config.protocol, 'model': config.model,
                        'selection': selection, 'intended_tasks': len(tasks), 'completed_tasks': len(episodes),
                        'split_coverage': len(episodes) / selection.get('split_count', len(tasks)),
                        'subset': selection.get('subset', False), 'wall_seconds': prior_wall + time.perf_counter() - started,
                        'completed_episode_model_calls': calls, 'total_legal_clicks': clicks,
                        'submitted_model_calls': sum(r['attempt'] == 1 for r in records),
                        'invalid_decision_rate': sum(failure_counts[k] for k in ('invalid_link', 'invalid_page', 'invalid_response')) / calls if calls else None,
                        'failure_counts': failure_counts,
                        'invalid_episode_rate': sum(failure_counts[k] for k in ('invalid_link', 'invalid_page', 'invalid_response')) / len(episodes) if episodes else None,
                        'mean_remote_call_latency_ms': usage['remote_latency_ms'] / len(records) if records else None,
                        'mean_click_latency_ms': metrics['total_inference_latency_ms'] / clicks if clicks else None,
                        'known_cost_per_episode': usage['known_cost'] / len(episodes) if episodes else None,
                        'known_cost_per_solved_task': usage['known_cost'] / successes if successes else None,
                        'run_dir': str(folder)})
        write_json_atomic(folder / 'metrics.json', metrics)
        return metrics

    write_status()
    try:
        evaluate_policy(PositionedPolicy(), graph, tasks[len(episodes):], config.max_steps,
                        start_index=len(episodes), on_task_start=task_start,
                        on_transition=transition, on_episode=episode_done, on_rejected=rejected)
        status = 'complete'
    except (Exception, KeyboardInterrupt) as exc:
        status = 'incomplete'
        error_info = {'kind': getattr(exc, 'kind', 'interrupted' if isinstance(exc, KeyboardInterrupt) else 'runtime'),
                      'message': str(exc) if isinstance(exc, RemoteError) else type(exc).__name__}
    finally:
        write_status()
        metrics = save_metrics()
    return metrics
