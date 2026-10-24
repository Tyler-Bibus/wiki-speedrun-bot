"""One-command, resumable remote lineup plus verified local checkpoint comparison."""
from dataclasses import asdict, replace
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import shutil
from threading import Event, RLock
from types import SimpleNamespace

from .comparison import write_comparison, validate_run
from .data.load import load_graph
from .remote import (append, make_client, make_policy, read_rows, run_remote, selected_tasks,
                     source_identity, usage_metrics)
from .remote_config import load_remote_config
from .training.checkpoints import object_fingerprint, sha256_file, write_json_atomic


MODELS = {'minimax': 'configs/minimax.toml', 'deepseek': 'configs/llm.toml', 'jev': 'configs/jev.toml'}
LOCAL = {'kev': 'checkpoints/comparison-untuned-kev-0.8b-bf75a6a',
         'sft': 'checkpoints/20261022T025733.517853Z-sft/update-001700',
         'rl': 'checkpoints/20261022T132731.513046Z-rl/update-000075'}


def local_hashes(path):
    return {name: sha256_file(Path(path) / name)
            for name in ('head.pt', 'adapter_model.safetensors', 'adapter_config.json', 'tokenizer.json', 'tokenizer_config.json')}


def import_local(source, destination, config, tasks, selection, tokenizer, checkpoint, graph):
    """Normalize an existing local run only after checking its actual evidence."""
    from .evaluation import summarize
    from .training.checkpoints import graph_fingerprint
    source, destination = Path(source), Path(destination)
    old_provenance = json.loads((source / 'provenance.json').read_text())
    old_config = json.loads((source / 'config.json').read_text())
    old_metrics = json.loads((source / 'metrics.json').read_text())
    if (source / 'error.json').exists() or (source / 'interrupted.json').exists():
        raise ValueError('Cannot reuse interrupted local evaluation')
    if old_provenance['data_hashes']['graph:sha256'] != graph_fingerprint(graph):
        raise ValueError('Local run graph mismatch')
    model = old_provenance['model']
    warm = model['warm_start']
    hashes = local_hashes(checkpoint)
    if hashes['head.pt'] != warm['head_sha256'] or hashes['adapter_model.safetensors'] != warm['weights_sha256']:
        raise ValueError('Local checkpoint hashes mismatch')
    choices = old_config['choices']
    if any(choices[k] != getattr(config, k) for k in ('node_max_choices', 'max_state_tokens', 'max_row_tokens', 'excerpt_chars')) or old_config['rl']['max_steps'] != config.max_steps:
        raise ValueError('Local run protocol settings mismatch')
    if model['kev_revision'] != tokenizer['kev_revision']:
        raise ValueError('Local encoder revision mismatch')
    if old_metrics['split'] != selection['split']:
        raise ValueError('Local split mismatch')
    task_hashes = [v for k, v in old_provenance['data_hashes'].items() if Path(k).name == selection['split'] + '_tasks.json']
    if task_hashes != [selection['tasks_file_sha256']]:
        raise ValueError('Local task file hash mismatch')
    all_episodes = read_rows(source / 'episodes.jsonl')
    expected_all = json.loads(Path(getattr(config, selection['split'] + '_tasks')).read_text())
    original_limit = old_provenance.get('selection', {}).get('max_tasks')
    if original_limit is not None:
        expected_all = expected_all[:original_limit]
    if len(all_episodes) < len(tasks) or [(e['start'], e['target']) for e in all_episodes] != [(e['start'], e['target']) for e in expected_all]:
        raise ValueError('Existing local run is missing/duplicating/reordering split pairs')
    episodes = all_episodes[:len(tasks)]
    episode_ids = {e['episode_id'] for e in episodes}
    transitions = [r for r in read_rows(source / 'transitions.jsonl') if r['episode_id'] in episode_ids]
    destination.mkdir(parents=True, exist_ok=False)
    for e in episodes:
        append(destination / 'episodes.jsonl', e)
    for row in transitions:
        append(destination / 'transitions.jsonl', row)
    values = asdict(config)
    values['model'] = str(checkpoint)
    provenance = {**source_identity(), 'protocol': 'matched_kev_tournament',
                  'requested_model': str(checkpoint), 'graph_sha256': graph_fingerprint(graph),
                  'ordered_tasks_sha256': object_fingerprint([asdict(t) for t in tasks]),
                  'selection': selection, 'tokenizer': tokenizer, 'checkpoint_sha256': hashes,
                  'imported_from': str(source), 'original_provenance': old_provenance}
    metrics = summarize(episodes, graph)
    metrics.update(status='complete', intended_tasks=len(tasks), model=str(checkpoint))
    write_json_atomic(destination / 'config.json', values)
    write_json_atomic(destination / 'provenance.json', provenance)
    write_json_atomic(destination / 'metrics.json', metrics)
    write_json_atomic(destination / 'status.json', {'status': 'complete'})
    validate_run(destination, graph)
    return destination


def local_result(name, output, config, tasks, selection, tokenizer, graph, *,
                 checkpoints=None, training_config=None):
    checkpoint = (LOCAL if checkpoints is None else checkpoints)[name]
    destination = output / (name + '-' + selection['split'])
    if destination.exists():
        validate_run(destination, graph)
        old = json.loads((destination / 'provenance.json').read_text())
        if old['checkpoint_sha256'] != local_hashes(checkpoint) or old['selection'] != selection:
            raise ValueError('Cached local comparison identity mismatch')
        return destination
    # Try existing immutable evaluations; import_local validates all evidence before reuse.
    for source in sorted(Path(config.run_dir).glob('*-evaluate')):
        try:
            if not (source / 'metrics.json').exists():
                continue
            m = json.loads((source / 'metrics.json').read_text())
            if m.get('split') != selection['split']:
                continue
            p = json.loads((source / 'provenance.json').read_text())
            if p.get('model', {}).get('warm_start', {}).get('head_sha256') != local_hashes(checkpoint)['head.pt']:
                continue
            return import_local(source, destination, config, tasks, selection, tokenizer, checkpoint, graph)
        except (ValueError, KeyError, OSError, StopIteration) as exc:
            print(f'Local artifact cannot be reused: {source.name}: {type(exc).__name__}', flush=True)
            if destination.exists():
                shutil.rmtree(destination)
    from .training.config import load_training_config
    from .training.rl import evaluate_trained
    c = load_training_config(Path(training_config or 'configs/rl.toml'))
    metrics = evaluate_trained(c, checkpoint, split=selection['split'], max_tasks=selection['max_tasks'])
    return import_local(metrics['run_dir'], destination, config, tasks, selection, tokenizer, checkpoint, graph)


def probe(config, output, name, snapshots, budget=None):
    from .remote_policy import LLMPolicy
    from .models.openrouter import strict_json, RemoteError
    folder = output / 'probes' / name
    folder.mkdir(parents=True, exist_ok=True)
    previous = folder / 'probe.json'
    client = make_client(config)
    client.resolved_model = snapshots.get(name)
    if previous.exists():
        record = json.loads(previous.read_text())
        if record.get('status') == 'complete':
            client.resolved_model = record['resolved_model']
            snapshots[name] = client.resolved_model
            return client
    def record_attempt(row):
        append(folder / 'requests.jsonl', row)
        if budget:
            budget.observe(row)
    client.on_attempt = record_attempt
    client.before_attempt = budget.check if budget else None
    try:
        if config.protocol == 'matched_kev_tournament':
            result = client.smoke_test()
            correct = result['choice'] == 'earth'
        else:
            result = client.complete(LLMPolicy(client, config).request({
                'current_article': 'Earth', 'target_article': 'Moon', 'remaining_clicks': 1,
                'outgoing_articles': [{'id': 'Moon', 'title': 'Moon'}, {'id': 'Mars', 'title': 'Mars'}]}))
            try:
                correct = result['finish_reason'] == 'stop' and strict_json(result['content']) == {'next_article': 'Moon'}
            except (ValueError, TypeError):
                correct = False
        if not correct:
            raise RemoteError('smoke_failed', 'Known-answer model probe failed')
        snapshots[name] = client.resolved_model
        write_json_atomic(previous, {'status': 'complete', 'resolved_model': client.resolved_model,
                                    'result': client.redact(result),
                                    'accounting': usage_metrics(read_rows(folder / 'requests.jsonl'))})
    except RemoteError as exc:
        write_json_atomic(previous, {'status': 'incomplete', 'error': {'kind': exc.kind, 'message': str(exc)},
                                    'accounting': usage_metrics(read_rows(folder / 'requests.jsonl'))})
        raise
    finally:
        client.on_attempt = None
        client.before_attempt = None
    return client


def benchmark_command(args):
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'benchmark.lock').open('a+') as lease:
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError('Another benchmark already owns this output directory') from None
        return _benchmark_command(args)


def _benchmark_command(args):
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if len(set(args.models)) != len(args.models) or len(set(args.splits)) != len(args.splits):
        raise ValueError('Models and splits must be unique')
    if args.max_tasks is not None and args.max_tasks < 1:
        raise ValueError('--max-tasks must be positive')
    workers = getattr(args, 'remote_workers', 3)
    if type(workers) is not int or workers < 1:
        raise ValueError('--remote-workers must be positive')
    episode_workers = getattr(args, 'workers_per_model', 1)
    if type(episode_workers) is not int or episode_workers < 1:
        raise ValueError('--workers-per-model must be positive')
    training_path = getattr(args, 'training_config', None)
    training = None
    if training_path:
        from .training.config import load_training_config
        training = load_training_config(training_path)
    checkpoints = dict(LOCAL)
    if getattr(args, 'sft_checkpoint', None):
        from .training.checkpoints import resolve_checkpoint
        checkpoints['sft'] = str(resolve_checkpoint(args.sft_checkpoint))

    def remote_config(name):
        config = replace(load_remote_config(MODELS[name]), run_dir=str(output / 'runs'))
        if args.credentials_file:
            config = replace(config, credentials_file=str(args.credentials_file))
        if training:
            config = replace(config, data_dir=training.data_dir,
                             dev_tasks=training.data.dev_tasks, test_tasks=training.data.test_tasks,
                             train_tasks=training.data.rl_train_tasks,
                             split_manifest=training.data.split_manifest,
                             max_steps=training.rl.max_steps,
                             excerpt_chars=training.choices.excerpt_chars,
                             node_max_choices=training.choices.node_max_choices,
                             max_state_tokens=training.choices.max_state_tokens,
                             max_row_tokens=training.choices.max_row_tokens)
        return config
    state_path = output / 'benchmark.json'
    settings = {'models': args.models, 'splits': args.splits, 'max_tasks': args.max_tasks,
                'include_local': args.include_local, 'checkpoints': checkpoints,
                'configs_sha256': {name: sha256_file(Path(MODELS[name])) for name in args.models}}
    if training_path:
        settings['training_config_sha256'] = sha256_file(Path(training_path))
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state['settings'] != settings:
            raise ValueError('Benchmark settings changed; choose a new --output directory')
    else:
        state = {'settings': settings, 'runs': {}, 'errors': {}, 'resolved_models': {}}
    from .spending import SuiteBudget
    budget = SuiteBudget(output, getattr(args, 'max_total_cost', 9.0))
    graph = load_graph(Path(training.data_dir if training else 'data'))
    configs = {name: remote_config(name) for name in args.models}
    # Validate every selected split before starting any paid requests.
    selections = {(name, split): selected_tasks(configs[name], split, max_tasks=args.max_tasks)
                  for name in args.models for split in args.splits}
    complete_by_split = {split: {} for split in args.splits}
    state_lock = RLock()
    state['errors'].pop('benchmark', None)
    cancelled = Event()
    def before_attempt():
        if cancelled.is_set():
            from .models.openrouter import RemoteError
            raise RemoteError('interrupted', 'Stopped by user')
        budget.check()
    guarded_budget = SimpleNamespace(check=before_attempt, observe=budget.observe)
    state['execution'] = {'remote_workers': min(workers, len(args.models)),
                          'workers_per_model': episode_workers,
                          'local_workers': 1}
    write_json_atomic(state_path, state)

    def evaluate_model(name):
        # Each worker owns its client, policy, pacing, and ordered episode files.
        config = configs[name]
        with state_lock:
            snapshots = dict(state['resolved_models'])
        for split in args.splits:
            label = name + '-' + split
            tasks, selection = selections[name, split]
            with state_lock:
                previous = state['runs'].get(label)
            try:
                if previous and json.loads((Path(previous) / 'status.json').read_text())['status'] == 'complete':
                    validate_run(previous, graph)
                    with state_lock:
                        complete_by_split[split][name] = previous
                        state['errors'].pop(label, None)
                        write_json_atomic(state_path, state)
                    continue
                if cancelled.is_set():
                    raise RuntimeError('Stopped by user')
                client = probe(config, output, name, snapshots, guarded_budget)
                with state_lock:
                    if name in snapshots:
                        state['resolved_models'][name] = snapshots[name]
                        write_json_atomic(state_path, state)
                policy, tokenizer = make_policy(config, client)
                def remember_run(folder):
                    with state_lock:
                        state['runs'][label] = str(folder)
                        write_json_atomic(state_path, state)
                parallel = episode_workers > 1 or previous and any(
                    (Path(previous) / name).exists() for name in ('parallel.json', 'execution-upgrades.json'))
                if parallel:
                    from .remote_parallel import run_remote_parallel
                    def worker_factory():
                        worker_client = make_client(config)
                        worker_client.resolved_model = client.resolved_model
                        worker_policy, worker_tokenizer = make_policy(config, worker_client)
                        return worker_client, worker_policy, worker_tokenizer
                    metrics = run_remote_parallel(config, tasks, selection, workers=episode_workers,
                                                  graph=graph, client=client, tokenizer_identity=tokenizer,
                                                  resume=previous, on_created=remember_run,
                                                  before_attempt_guard=before_attempt, on_attempt_observer=budget.observe,
                                                  worker_factory=worker_factory, cancel_event=cancelled)
                else:
                    metrics = run_remote(config, tasks, selection, graph=graph, client=client,
                                         policy=policy, tokenizer_identity=tokenizer, resume=previous, on_created=remember_run,
                                         before_attempt_guard=before_attempt, on_attempt_observer=budget.observe)
                with state_lock:
                    state['runs'][label] = metrics['run_dir']
                    if metrics['status'] == 'complete':
                        complete_by_split[split][name] = metrics['run_dir']
                        state['errors'].pop(label, None)
                    else:
                        state['errors'][label] = json.loads((Path(metrics['run_dir']) / 'status.json').read_text())['error']
            except (ValueError, RuntimeError, OSError, ImportError) as exc:
                with state_lock:
                    state['errors'][label] = {'kind': getattr(exc, 'kind', type(exc).__name__),
                                              'message': str(exc) if hasattr(exc, 'kind') else type(exc).__name__}
            with state_lock:
                # A failed development split blocks test only for this model.
                failed = label in state['errors']
                if failed:
                    for skipped_split in args.splits[args.splits.index(split) + 1:]:
                        state['errors'][name + '-' + skipped_split] = {'kind': 'setup_dependency', 'message': 'Earlier split did not finish'}
                    print(f'{label}: incomplete: {state["errors"][label]}', flush=True)
                write_json_atomic(state_path, state)
            if failed:
                break

    if args.models:
        pool = ThreadPoolExecutor(max_workers=min(workers, len(args.models)))
        try:
            list(pool.map(evaluate_model, args.models))
        except KeyboardInterrupt:
            cancelled.set()
            with state_lock:
                state['errors']['benchmark'] = {'kind': 'interrupted', 'message': 'Stopped by user'}
            print('Stopping new requests; saving results from requests already in flight.', flush=True)
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
    # GPU evaluations are deliberately sequential and start after remote workers finish.
    if args.include_local and not cancelled.is_set():
        from .models.kev_budget import KevContextBudgeter
        config = remote_config('jev')
        tokenizer = KevContextBudgeter(config.tokenizer, config).metadata
        for split in args.splits:
            tasks, selection = selected_tasks(config, split, max_tasks=args.max_tasks)
            for name in checkpoints:
                label = name + '-' + split
                try:
                    result = local_result(name, output, config, tasks, selection, tokenizer, graph,
                                          checkpoints=checkpoints, training_config=training_path)
                    state['runs'][label] = str(result)
                    complete_by_split[split][name] = str(result)
                    state['errors'].pop(label, None)
                except (ValueError, RuntimeError, OSError, ImportError) as exc:
                    state['errors'][label] = {'kind': type(exc).__name__, 'message': str(exc)}
                write_json_atomic(state_path, state)
    order = args.models + (list(checkpoints) if args.include_local else [])
    for split, results in complete_by_split.items():
        folders = [results[name] for name in order if name in results]
        if folders:
            write_comparison(folders, output / (split + '-results.md'), allow_system_protocols=True, graph=graph)
    lines = ['# OpenRouter benchmark execution', '',
             'Rerun the same command to resume completed episode boundaries. Model failures remain in the denominator.', '',
             'Requested settings: `' + json.dumps(settings, sort_keys=True) + '`', '',
             'Historical SFT test results were viewed before this experiment. No test outcomes are used for checkpoint or prompt selection.', '']
    for split in args.splits:
        p = output / (split + '-results.md')
        if p.exists():
            lines += [p.read_text(), '']
    lines += ['## Incomplete/setup errors', '', '```json', json.dumps(state['errors'], indent=2), '```', '']
    probe_records = [row for p in (output / 'probes').glob('*/requests.jsonl') for row in read_rows(p)]
    lines += ['## Known-answer probe accounting', '', '```json', json.dumps(usage_metrics(probe_records), indent=2), '```', '']
    lines += ['## Shared suite budget', '', '```json', (output / 'budget.json').read_text(), '```', '']
    report = output / 'report.md'
    report.write_text('\n'.join(lines))
    print(f'Report: {report}', flush=True)
    return 1 if state['errors'] else 0
