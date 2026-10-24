#!/usr/bin/env python3
"""Record verified scheduling and provider-retry upgrades for remote runs, offline."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import uuid

from wiki_speedrun.benchmark import MODELS
from wiki_speedrun.remote import selected_tasks, source_identity
from wiki_speedrun.remote_config import RemoteConfig
from wiki_speedrun.remote_parallel import IDENTITY_KEYS, source_compatible
from wiki_speedrun.training.checkpoints import object_fingerprint, sha256_file, write_json_atomic


BASELINE = Path(__file__).with_name('parallel-remote-baseline.json')
CHANGED_FILES = {'src/wiki_speedrun/benchmark.py', 'src/wiki_speedrun/cli.py', 'src/wiki_speedrun/batch.py'}
NEW_FILES = {'src/wiki_speedrun/remote_parallel.py'}
TRANSPORT_UPGRADE = Path(__file__).with_name('provider-retry-upgrade.json')


def verify_scheduling_upgrade():
    baseline = json.loads(BASELINE.read_text())
    current = {str(p): sha256_file(p) for p in sorted(Path('src/wiki_speedrun').rglob('*.py'))}
    old = baseline['files']
    if object_fingerprint(old) != baseline['source_sha256']:
        raise ValueError('Scheduling baseline fingerprint is invalid')
    changed = {path for path in old if current.get(path) != old[path]}
    # Only the exact reviewed transport patch is compatible. Other transport or
    # policy changes must still fail closed, even when scheduling also changed.
    patch = json.loads(TRANSPORT_UPGRADE.read_text())
    approved = set()
    if (old.get(patch['file']) == patch['from_sha256']
            and current.get(patch['file']) == patch['to_sha256']):
        approved.add(patch['file'])
    if not changed <= CHANGED_FILES | approved or set(current) - set(old) != NEW_FILES:
        raise ValueError('Source changes include unreviewed non-scheduling code; upgrade refused')
    return baseline, source_identity()['source_sha256'], sorted(changed | NEW_FILES)


def enable_parallel(output, *, dry_run=False):
    baseline, target_hash, changed = verify_scheduling_upgrade()
    output = Path(output)
    state = json.loads((output / 'benchmark.json').read_text())
    planned = []
    for label, folder in state['runs'].items():
        name, _, split = label.rpartition('-')
        if name not in MODELS:
            continue
        folder = Path(folder)
        provenance = json.loads((folder / 'provenance.json').read_text())
        if source_compatible(folder, provenance, target_hash):
            continue
        if provenance['source_sha256'] != baseline['source_sha256']:
            raise ValueError(f'{label}: run predates the verified scheduling baseline')
        config = RemoteConfig(**json.loads((folder / 'config.json').read_text()))
        signature = {k: v for k, v in asdict(config).items()
                     if k not in ('max_requests', 'max_cost', 'credentials_file', 'run_dir')}
        if object_fingerprint(signature) != provenance['settings_sha256']:
            raise ValueError(f'{label}: original run config does not match its provenance')
        _, selection = selected_tasks(config, split, max_tasks=state['settings']['max_tasks'])
        if selection != provenance['selection']:
            raise ValueError(f'{label}: frozen task selection changed')
        row = {'from_source_sha256': provenance['source_sha256'], 'to_source_sha256': target_hash,
               'identity_sha256': object_fingerprint({key: provenance[key] for key in IDENTITY_KEYS}),
               'reason': 'Concurrent independent episodes; unchanged request, policy, navigation, and comparison code',
               'changed_source_files': changed}
        patch = json.loads(TRANSPORT_UPGRADE.read_text())
        if patch['file'] in changed:
            row['reason'] = ('Concurrent independent episodes and bounded retries for explicit provider generation failures; '
                             'unchanged request payloads, policy, navigation, and comparison code')
            row['reviewed_transport_upgrade'] = patch
        planned.append((folder, row))
    report = {'runs': [str(folder) for folder, _ in planned], 'to_source_sha256': target_hash, 'dry_run': dry_run}
    if dry_run or not planned:
        return report
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    backup = output / 'execution-upgrades' / (stamp + '-' + uuid.uuid4().hex[:8] + '.json')
    write_json_atomic(backup, {'benchmark': state, 'upgrade': report,
                              'original_provenance': {str(folder): json.loads((folder / 'provenance.json').read_text())
                                                      for folder, _ in planned}})
    for folder, row in planned:
        path = folder / 'execution-upgrades.json'
        saved = json.loads(path.read_text()) if path.exists() else {'upgrades': []}
        saved['upgrades'].append({**row, 'audit': str(backup)})
        write_json_atomic(path, saved)
    report['audit'] = str(backup)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    try:
        report = enable_parallel(args.output, dry_run=args.dry_run)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f'Error: {error}\n')
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
