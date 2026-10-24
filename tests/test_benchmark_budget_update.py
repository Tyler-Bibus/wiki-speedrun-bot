"""Budget repair preserves paid progress and all non-budget resume checks."""
from argparse import Namespace
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.update_benchmark_budgets import signature, update_budget_manifest
from wiki_speedrun.benchmark import benchmark_command
from wiki_speedrun.data.load import WikiGraph
from wiki_speedrun.data.tasks import Task
from wiki_speedrun.models.openrouter import OpenRouterChatClient
from wiki_speedrun.remote import source_identity
from wiki_speedrun.remote_config import RemoteConfig
from wiki_speedrun.training.checkpoints import object_fingerprint, sha256_file


class BudgetUpdateTests(unittest.TestCase):
    def write_config(self, path, **changes):
        values = {'model': 'fixture-model', 'protocol': 'llm_full_list_json_object',
                  'structured_output': 'json_object', 'max_requests': 1, 'max_cost': 1}
        values.update(changes)
        path.write_text(''.join(f'{key} = {json.dumps(value)}\n' for key, value in values.items()))

    def test_budget_increase_resumes_same_run_without_repeating_completed_episode(self):
        selection = {'split': 'dev', 'split_count': 2, 'max_tasks': None, 'subset': False}
        tasks = [Task('A', 'C'), Task('A', 'B')]
        graph = WikiGraph({'A', 'B', 'C'}, {'A': ['B', 'C'], 'B': [], 'C': []})
        calls = []
        def probe(config, output, name, snapshots, budget):
            def transport(request, timeout):
                article = 'C' if not calls else 'B'
                calls.append(article)
                return io.BytesIO(json.dumps({
                    'id': 'fixture', 'model': 'snapshot', 'provider': 'fixture', 'usage': {'cost': .01},
                    'choices': [{'message': {'content': json.dumps({'next_article': article})},
                                 'finish_reason': 'stop'}],
                }).encode())
            return OpenRouterChatClient('fixture-key', config.model, retries=0, transport=transport)

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config = root / 'model.toml'
            self.write_config(config)
            args = Namespace(output=root / 'output', max_tasks=None, models=['minimax'], splits=['dev'],
                             include_local=False, credentials_file=None, remote_workers=1, max_total_cost=9)
            with patch.dict('wiki_speedrun.benchmark.MODELS', {'minimax': str(config)}), \
                 patch('wiki_speedrun.benchmark.load_graph', return_value=graph), \
                 patch('wiki_speedrun.benchmark.selected_tasks', return_value=(tasks, selection)), \
                 patch('scripts.update_benchmark_budgets.selected_tasks', return_value=(tasks, selection)), \
                 patch('wiki_speedrun.benchmark.probe', side_effect=probe):
                self.assertEqual(benchmark_command(args), 1)
                state_path = args.output / 'benchmark.json'
                original_state = json.loads(state_path.read_text())
                folder = Path(original_state['runs']['minimax-dev'])
                prefix = (folder / 'episodes.jsonl').read_bytes()
                originals = {name: (folder / name).read_bytes() for name in ('config.json', 'provenance.json')}
                self.write_config(config, max_requests=10, max_cost=10)
                with self.assertRaisesRegex(ValueError, 'Benchmark settings changed'):
                    benchmark_command(args)
                self.assertEqual(calls, ['C'])
                report = update_budget_manifest(args.output, dry_run=True)
                self.assertIn('minimax', report['changes'])
                self.assertEqual(json.loads(state_path.read_text()), original_state)
                report = update_budget_manifest(args.output)
                backup = json.loads(Path(report['backup']).read_text())
                self.assertEqual(backup['previous_benchmark'], original_state)
                self.assertEqual((folder / 'episodes.jsonl').read_bytes(), prefix)
                for name, value in originals.items():
                    self.assertEqual((folder / name).read_bytes(), value)
                self.assertEqual(benchmark_command(args), 0)
                self.assertEqual(calls, ['C', 'B'])
                state = json.loads(state_path.read_text())
                self.assertEqual(state['runs']['minimax-dev'], str(folder))
                self.assertTrue((folder / 'episodes.jsonl').read_bytes().startswith(prefix))
                metrics = json.loads((folder / 'metrics.json').read_text())
                self.assertEqual(metrics['completed_tasks'], 2)
                self.assertEqual(metrics['http_attempts'], 2)
                self.assertAlmostEqual(metrics['known_cost'], .02)

    def fixture(self, root):
        config_path = root / 'model.toml'
        self.write_config(config_path)
        folder = root / 'runs' / 'fixture'
        folder.mkdir(parents=True)
        config = RemoteConfig(model='fixture-model', protocol='llm_full_list_json_object',
                              structured_output='json_object', run_dir=str(root / 'runs'),
                              max_requests=1, max_cost=1)
        saved = asdict(config)
        selection = {'split': 'dev', 'max_tasks': None, 'subset': False, 'split_count': 1}
        provenance = {'source_sha256': source_identity()['source_sha256'], 'selection': selection,
                      'settings_sha256': object_fingerprint(signature(saved))}
        (folder / 'config.json').write_text(json.dumps(saved))
        (folder / 'provenance.json').write_text(json.dumps(provenance))
        state = {'settings': {'models': ['minimax'], 'splits': ['dev'], 'max_tasks': None,
                             'configs_sha256': {'minimax': sha256_file(config_path)}},
                 'runs': {'minimax-dev': str(folder)}, 'errors': {}}
        (root / 'benchmark.json').write_text(json.dumps(state))
        return config_path, folder, selection

    def test_protocol_source_selection_and_budget_decreases_are_rejected_atomically(self):
        for case in ('model', 'temperature', 'source', 'selection', 'lower-limit', 'dataset'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as td:
                root = Path(td)
                config, folder, selection = self.fixture(root)
                changes = {'max_requests': 10, 'max_cost': 10}
                if case == 'model':
                    changes['model'] = 'other-model'
                if case == 'temperature':
                    changes['temperature'] = 1
                if case == 'source':
                    p = folder / 'provenance.json'
                    d = json.loads(p.read_text()); d['source_sha256'] = 'other-source'
                    p.write_text(json.dumps(d))
                if case == 'selection':
                    selection = selection | {'split_count': 2}
                if case == 'lower-limit':
                    (folder / 'sessions.jsonl').write_text(json.dumps({'max_requests': 20, 'max_cost': 20}) + '\n')
                self.write_config(config, **changes)
                original = (root / 'benchmark.json').read_bytes()
                with patch.dict('wiki_speedrun.benchmark.MODELS', {'minimax': str(config)}), \
                     patch('scripts.update_benchmark_budgets.selected_tasks', return_value=([], selection)):
                    with self.assertRaises(ValueError):
                        update_budget_manifest(root, Path('configs/snap-full.toml') if case == 'dataset' else None)
                self.assertEqual((root / 'benchmark.json').read_bytes(), original)
                self.assertFalse((root / 'budget-updates').exists())


if __name__ == '__main__':
    unittest.main()
