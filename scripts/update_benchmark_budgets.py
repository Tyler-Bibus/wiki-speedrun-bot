#!/usr/bin/env python3
"""Accept verified budget-only config increases for an existing benchmark, offline.

Evaluation code, original run configs/provenance, and paid traces stay unchanged.
The prior suite manifest is backed up before its config-file hashes are updated.
"""
import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from wiki_speedrun.benchmark import MODELS
from wiki_speedrun.remote import read_rows, selected_tasks, source_identity
from wiki_speedrun.remote_config import load_remote_config
from wiki_speedrun.training.checkpoints import object_fingerprint, sha256_file, write_json_atomic
from wiki_speedrun.training.config import load_training_config


LIMITS = ('max_requests', 'max_cost')
RUNTIME_FIELDS = (*LIMITS, 'credentials_file', 'run_dir')


def signature(config):
    return {key: value for key, value in config.items() if key not in RUNTIME_FIELDS}


def update_budget_manifest(output, training_config=None, *, dry_run=False):
    output = Path(output)
    path = output / 'benchmark.json'
    original_bytes = path.read_bytes()
    state = json.loads(original_bytes)
    settings = state['settings']
    training = None
    expected_training_hash = settings.get('training_config_sha256')
    if expected_training_hash is not None:
        if training_config is None or sha256_file(Path(training_config)) != expected_training_hash:
            raise ValueError('Supply the unchanged --training-config used for this benchmark')
        training = load_training_config(training_config)
    elif training_config is not None:
        raise ValueError('Cannot add a different dataset to an existing benchmark')

    source_hash = source_identity()['source_sha256']
    changes = {}
    for name in settings['models']:
        config_path = Path(MODELS[name])
        new_hash = sha256_file(config_path)
        old_hash = settings['configs_sha256'][name]
        if old_hash == new_hash:
            continue
        config = replace(load_remote_config(config_path), run_dir=str(output / 'runs'))
        if training is not None:
            config = replace(config, data_dir=training.data_dir,
                             dev_tasks=training.data.dev_tasks, test_tasks=training.data.test_tasks,
                             train_tasks=training.data.rl_train_tasks,
                             split_manifest=training.data.split_manifest,
                             max_steps=training.rl.max_steps,
                             excerpt_chars=training.choices.excerpt_chars,
                             node_max_choices=training.choices.node_max_choices,
                             max_state_tokens=training.choices.max_state_tokens,
                             max_row_tokens=training.choices.max_row_tokens)
        current = asdict(config)
        checked = []
        for split in settings['splits']:
            folder = state['runs'].get(name + '-' + split)
            if folder is None:
                continue
            folder = Path(folder)
            old = json.loads((folder / 'config.json').read_text())
            provenance = json.loads((folder / 'provenance.json').read_text())
            old_signature = signature(old)
            if object_fingerprint(old_signature) != provenance['settings_sha256']:
                raise ValueError(f'{name}: saved run config does not match its provenance')
            if signature(current) != old_signature:
                differing = sorted(key for key in set(current) | set(old)
                                   if key not in RUNTIME_FIELDS and current.get(key) != old.get(key))
                raise ValueError(f'{name}: non-budget settings changed: {differing}')
            from wiki_speedrun.remote_parallel import source_compatible
            if not source_compatible(folder, provenance, source_hash):
                raise ValueError(f'{name}: evaluation source identity changed; budget migration refused')
            _, selection = selected_tasks(config, split, max_tasks=settings['max_tasks'])
            if selection != provenance['selection']:
                raise ValueError(f'{name}: frozen task selection changed')
            previous_limits = {key: old[key] for key in LIMITS}
            for session in read_rows(folder / 'sessions.jsonl'):
                for key in LIMITS:
                    if key in session:
                        previous_limits[key] = max(previous_limits[key], session[key])
            if any(current[key] < previous_limits[key] for key in LIMITS):
                raise ValueError(f'{name}: this repair only permits budget increases')
            checked.append({'run': str(folder), 'old_limits': previous_limits})
        if not checked:
            raise ValueError(f'{name}: no saved run exists to verify the original protocol')
        changes[name] = {'old_config_sha256': old_hash, 'new_config_sha256': new_hash,
                         'new_limits': {key: current[key] for key in LIMITS}, 'verified_runs': checked}

    report = {'changes': changes, 'source_sha256': source_hash, 'dry_run': dry_run}
    if dry_run or not changes:
        return report
    if path.read_bytes() != original_bytes:
        raise RuntimeError('Benchmark state changed during verification; stop its worker and retry')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    backup = output / 'budget-updates' / (stamp + '-' + uuid.uuid4().hex[:8] + '.json')
    write_json_atomic(backup, {'previous_benchmark': state, 'verification': report})
    for name, change in changes.items():
        settings['configs_sha256'][name] = change['new_config_sha256']
    state.setdefault('budget_updates', []).append({'backup': str(backup),
                                                  'changes': changes, 'source_sha256': source_hash})
    write_json_atomic(path, state)
    report['backup'] = str(backup)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--training-config', type=Path)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        report = update_budget_manifest(args.output, args.training_config, dry_run=args.dry_run)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        parser.exit(1, f'Error: {error}\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
