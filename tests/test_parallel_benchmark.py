"""Offline concurrency checks using overlapping fake HTTP requests."""
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
import io
import json
from pathlib import Path
import tempfile
from threading import Barrier, Lock, get_ident
import unittest
from unittest.mock import patch

from wiki_speedrun.benchmark import benchmark_command
from wiki_speedrun.data.load import WikiGraph
from wiki_speedrun.data.tasks import Task
from wiki_speedrun.models.openrouter import OpenRouterChatClient, RemoteError
from wiki_speedrun.spending import SuiteBudget


def graph():
    return WikiGraph({'A', 'B', 'C'}, {'A': ['B', 'C'], 'B': [], 'C': []})


def tasks(config, split, max_tasks=None):
    return [Task('A', 'C')], {'split': split, 'split_count': 1,
                             'max_tasks': max_tasks, 'subset': max_tasks is not None}


def response():
    return io.BytesIO(json.dumps({
        'id': 'fixture', 'model': 'snapshot', 'provider': 'fixture',
        'usage': {'cost': .01},
        'choices': [{'message': {'content': '{"next_article":"C"}'}, 'finish_reason': 'stop'}],
    }).encode())


class ParallelBenchmarkTests(unittest.TestCase):
    def args(self, output, **kwargs):
        values = dict(output=Path(output), max_tasks=None, models=['minimax', 'deepseek'],
                      splits=['dev', 'test'], include_local=False, credentials_file=None,
                      remote_workers=2, max_total_cost=9)
        return Namespace(**(values | kwargs))

    def test_models_overlap_keep_split_order_and_reuse_completed_runs(self):
        rendezvous = Barrier(2)
        lock = Lock()
        events = []

        def probe(config, output, name, snapshots, budget):
            snapshots[name] = 'snapshot'
            def transport(request, timeout):
                with lock:
                    events.append(name)
                rendezvous.wait(timeout=5)
                return response()
            return OpenRouterChatClient('fixture-key', config.model, retries=0, transport=transport)

        with tempfile.TemporaryDirectory() as td:
            args = self.args(td)
            with patch('wiki_speedrun.benchmark.load_graph', return_value=graph()), \
                 patch('wiki_speedrun.benchmark.selected_tasks', side_effect=tasks), \
                 patch('wiki_speedrun.benchmark.probe', side_effect=probe) as probes:
                self.assertEqual(benchmark_command(args), 0)
                # Both models entered transport before either could complete a request.
                self.assertEqual(set(events[:2]), set(args.models))
                self.assertEqual(len(events), 4)
                first_calls = probes.call_count
                self.assertEqual(benchmark_command(args), 0)
                self.assertEqual(probes.call_count, first_calls)
            state = json.loads((Path(td) / 'benchmark.json').read_text())
            self.assertEqual(set(state['runs']), {'minimax-dev', 'minimax-test',
                                                 'deepseek-dev', 'deepseek-test'})
            self.assertEqual(state['errors'], {})
            self.assertEqual(state['resolved_models'], {'minimax': 'snapshot', 'deepseek': 'snapshot'})
            for name in args.models:
                dev = Path(state['runs'][name + '-dev'])
                test = Path(state['runs'][name + '-test'])
                self.assertLess(dev.name, test.name)
            budget = json.loads((Path(td) / 'budget.json').read_text())
            self.assertEqual(budget['http_attempts'], 4)
            self.assertAlmostEqual(budget['known_cost'], .04)

    def test_failed_model_does_not_block_other_models_test_split(self):
        failures = {'minimax'}
        calls = {'minimax': 0, 'deepseek': 0}
        def probe(config, output, name, snapshots, budget):
            def transport(request, timeout):
                calls[name] += 1
                if name in failures:
                    raise RemoteError('provider_unavailable', 'fixture outage')
                return response()
            return OpenRouterChatClient('fixture-key', config.model, retries=0, transport=transport)

        with tempfile.TemporaryDirectory() as td:
            with patch('wiki_speedrun.benchmark.load_graph', return_value=graph()), \
                 patch('wiki_speedrun.benchmark.selected_tasks', side_effect=tasks), \
                 patch('wiki_speedrun.benchmark.probe', side_effect=probe):
                self.assertEqual(benchmark_command(self.args(td)), 1)
            state = json.loads((Path(td) / 'benchmark.json').read_text())
            self.assertEqual(state['errors']['minimax-dev']['kind'], 'provider_unavailable')
            self.assertEqual(state['errors']['minimax-test']['kind'], 'setup_dependency')
            self.assertNotIn('deepseek-test', state['errors'])
            metrics = json.loads((Path(state['runs']['deepseek-test']) / 'metrics.json').read_text())
            self.assertEqual(metrics['status'], 'complete')
            failures.clear()
            with patch('wiki_speedrun.benchmark.load_graph', return_value=graph()), \
                 patch('wiki_speedrun.benchmark.selected_tasks', side_effect=tasks), \
                 patch('wiki_speedrun.benchmark.probe', side_effect=probe):
                self.assertEqual(benchmark_command(self.args(td)), 0)
            self.assertEqual(calls, {'minimax': 3, 'deepseek': 2})
            restored = json.loads((Path(td) / 'budget.json').read_text())
            self.assertEqual(restored['http_attempts'], 5)
            self.assertEqual(restored['unknown_cost_attempts'], 1)
            self.assertAlmostEqual(restored['known_cost'], .04)

    def test_full_snap_selection_and_local_checkpoint_override_stay_sequential(self):
        main_thread = get_ident()
        seen_configs = []
        local_calls = []
        def selected(config, split, max_tasks=None):
            seen_configs.append(config)
            return tasks(config, split, max_tasks)
        def probe(config, output, name, snapshots, budget):
            return OpenRouterChatClient('fixture-key', config.model, retries=0,
                                        transport=lambda request, timeout: response())
        def local(name, output, config, *args, checkpoints, training_config):
            self.assertEqual(get_ident(), main_thread)
            local_calls.append((name, checkpoints, training_config))
            return output / ('fixture-' + name)

        with tempfile.TemporaryDirectory() as td:
            checkpoint = Path(td) / 'new-checkpoint'
            checkpoint.mkdir()
            args = self.args(td, models=['deepseek'], include_local=True, remote_workers=1,
                             training_config=Path('configs/snap-full.toml'), sft_checkpoint=checkpoint)
            with patch('wiki_speedrun.benchmark.load_graph', return_value=graph()), \
                 patch('wiki_speedrun.benchmark.selected_tasks', side_effect=selected), \
                 patch('wiki_speedrun.benchmark.probe', side_effect=probe), \
                 patch('wiki_speedrun.benchmark.local_result', side_effect=local), \
                 patch('wiki_speedrun.benchmark.write_comparison'), \
                 patch('wiki_speedrun.models.kev_budget.KevContextBudgeter'):
                self.assertEqual(benchmark_command(args), 0)
            self.assertTrue(seen_configs)
            for config in seen_configs:
                self.assertEqual(config.dev_tasks, 'data/splits/snap-full/dev_tasks.json')
                self.assertEqual(config.test_tasks, 'data/splits/snap-full/test_tasks.json')
                self.assertEqual(config.split_manifest, 'data/splits/snap-full/manifest.json')
            self.assertEqual([v[0] for v in local_calls], ['kev', 'sft', 'rl'] * 2)
            for _, checkpoints, path in local_calls:
                self.assertEqual(checkpoints['sft'], str(checkpoint))
                self.assertEqual(path, args.training_config)

    def test_shared_budget_retains_all_concurrent_receipts(self):
        with tempfile.TemporaryDirectory() as td:
            budget = SuiteBudget(td, .05)
            with ThreadPoolExecutor(max_workers=4) as pool:
                list(pool.map(lambda _: budget.observe({'response': {'usage': {'cost': .001}}}), range(100)))
            saved = json.loads((Path(td) / 'budget.json').read_text())
            self.assertEqual(saved['http_attempts'], 100)
            self.assertAlmostEqual(saved['known_cost'], .1)
            with self.assertRaisesRegex(RemoteError, 'limit reached'):
                budget.check()

    def test_multiple_episode_workers_per_model_reach_benchmark(self):
        barrier = Barrier(4)
        lock = Lock()
        calls = []
        starts = ['A0', 'A1', 'A2']
        fixture_graph = WikiGraph(set(starts) | {'T', 'X'},
                                 {s: ['T', 'X'] for s in starts} | {'T': [], 'X': []})
        fixture_tasks = [Task(s, 'T') for s in starts]
        def selected(config, split, max_tasks=None):
            return fixture_tasks, {'split': split, 'split_count': 3, 'max_tasks': None, 'subset': False}
        def create(config):
            def transport(request, timeout):
                data = json.loads(request.data)
                source = json.loads(data['messages'][1]['content'])['current_article']
                with lock:
                    calls.append((config.model, source))
                if source in ('A0', 'A1'):
                    barrier.wait(timeout=5)
                return io.BytesIO(json.dumps({
                    'id': 'fixture', 'model': 'snapshot', 'provider': 'fixture', 'usage': {'cost': .01},
                    'choices': [{'message': {'content': '{"next_article":"T"}'}, 'finish_reason': 'stop'}],
                }).encode())
            return OpenRouterChatClient('fixture-key', config.model, retries=0, transport=transport)
        with tempfile.TemporaryDirectory() as td:
            args = self.args(td, splits=['dev'], workers_per_model=2)
            with patch('wiki_speedrun.benchmark.load_graph', return_value=fixture_graph), \
                 patch('wiki_speedrun.benchmark.selected_tasks', side_effect=selected), \
                 patch('wiki_speedrun.benchmark.probe', side_effect=lambda config, *args: create(config)), \
                 patch('wiki_speedrun.benchmark.make_client', side_effect=create):
                self.assertEqual(benchmark_command(args), 0)
            self.assertEqual(len(calls), 6)
            self.assertEqual(len(set(calls)), 6)
            state = json.loads((Path(td) / 'benchmark.json').read_text())
            self.assertEqual(state['execution']['workers_per_model'], 2)
            self.assertEqual(state['errors'], {})

    def test_invalid_worker_count_fails_before_requests(self):
        with tempfile.TemporaryDirectory() as td, patch('wiki_speedrun.benchmark.probe') as probe:
            with self.assertRaisesRegex(ValueError, '--remote-workers'):
                benchmark_command(self.args(td, remote_workers=0))
            probe.assert_not_called()


if __name__ == '__main__':
    unittest.main()
