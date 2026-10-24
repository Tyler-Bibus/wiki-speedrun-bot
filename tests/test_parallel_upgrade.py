"""Reviewed execution upgrades retain provenance and reject unknown code changes."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.enable_parallel_remote import BASELINE, enable_parallel, verify_scheduling_upgrade
import test_parallel_episodes as fixtures
from wiki_speedrun.remote_parallel import run_remote_parallel, source_compatible
from wiki_speedrun.remote import source_identity
from wiki_speedrun.training.checkpoints import sha256_file


class ParallelUpgradeTests(unittest.TestCase):
    def test_legacy_source_upgrade_preserves_original_records_and_identity_checks(self):
        helper = fixtures.ParallelEpisodeTests()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config, graph, tasks, selection = helper.fixture(root / 'runs', count=1)
            factory = helper.factory(config, lambda request, timeout: fixtures.reply())
            result = run_remote_parallel(config, tasks, selection, workers=1, graph=graph, worker_factory=factory)
            folder = Path(result['run_dir'])
            path = folder / 'provenance.json'
            provenance = json.loads(path.read_text())
            provenance['source_sha256'] = json.loads(BASELINE.read_text())['source_sha256']
            path.write_text(json.dumps(provenance))
            originals = {name: (folder / name).read_bytes()
                         for name in ('episodes.jsonl', 'requests.jsonl', 'transitions.jsonl', 'provenance.json')}
            state = {'settings': {'models': ['minimax'], 'splits': ['dev'], 'max_tasks': None},
                     'runs': {'minimax-dev': str(folder)}}
            (root / 'benchmark.json').write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                run_remote_parallel(config, tasks, selection, workers=2, graph=graph, resume=folder, worker_factory=factory)
            with patch('scripts.enable_parallel_remote.selected_tasks', return_value=(tasks, selection)):
                report = enable_parallel(root, dry_run=True)
                self.assertEqual(report['runs'], [str(folder)])
                self.assertFalse((folder / 'execution-upgrades.json').exists())
                enable_parallel(root)
            self.assertTrue(source_compatible(folder, provenance, source_identity()['source_sha256']))
            for name, raw in originals.items():
                self.assertEqual((folder / name).read_bytes(), raw)
            result = run_remote_parallel(config, tasks, selection, workers=2, graph=graph, resume=folder, worker_factory=factory)
            self.assertEqual(result['http_attempts'], 1)
            for name, raw in originals.items():
                self.assertEqual((folder / name).read_bytes(), raw)
            changed = dict(provenance, settings_sha256='different-protocol')
            self.assertFalse(source_compatible(folder, changed, source_identity()['source_sha256']))
            self.assertFalse(source_compatible(folder, provenance, 'different-source'))

    def test_non_scheduling_source_changes_cannot_be_migrated(self):
        original = sha256_file
        def tampered(path):
            return 'changed-policy' if str(path).endswith('remote_policy.py') else original(path)
        with patch('scripts.enable_parallel_remote.sha256_file', side_effect=tampered):
            with self.assertRaisesRegex(ValueError, 'non-scheduling'):
                verify_scheduling_upgrade()

    def test_unreviewed_transport_changes_cannot_be_migrated(self):
        original = sha256_file
        def tampered(path):
            return 'different-transport' if str(path).endswith('models/openrouter.py') else original(path)
        with patch('scripts.enable_parallel_remote.sha256_file', side_effect=tampered):
            with self.assertRaisesRegex(ValueError, 'non-scheduling'):
                verify_scheduling_upgrade()


if __name__ == '__main__':
    unittest.main()
