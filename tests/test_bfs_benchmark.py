"""BFS timing covers the complete task list and preserves evaluation inputs."""

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.benchmark_bfs import main, navigate_search, time_bfs, time_navigation
from wiki_speedrun.data.load import WikiGraph
from wiki_speedrun.data.tasks import Task, shortest_path
from wiki_speedrun.training.checkpoints import object_fingerprint, sha256_file


class BfsBenchmarkTests(unittest.TestCase):
    def test_navigation_accepts_twelfth_move_but_never_takes_thirteenth(self):
        nodes = [str(i) for i in range(14)]
        graph = WikiGraph(set(nodes), {a: [b] for a, b in zip(nodes, nodes[1:])})
        for algorithm in ('bfs', 'dfs'):
            for target, success in [('12', True), ('13', False)]:
                with self.subTest(algorithm=algorithm, target=target):
                    e = navigate_search(graph, Task('0', target), 12, algorithm)
                    self.assertEqual(e['success'], success)
                    self.assertEqual(e['steps'], 12)
                    self.assertEqual(e['path'], nodes[:13])
                    self.assertEqual(e['termination_reason'], 'success' if success else 'step_limit')

    def test_backtracking_consumes_moves_and_can_exhaust_budget(self):
        graph = WikiGraph(set('ABCT'), {'A': ['B', 'C'], 'B': ['A'], 'C': ['T'], 'T': []})
        for algorithm in ('bfs', 'dfs'):
            with self.subTest(algorithm=algorithm):
                e = navigate_search(graph, Task('A', 'T'), 4, algorithm)
                self.assertEqual(e['path'], ['A', 'B', 'A', 'C', 'T'])
                self.assertEqual(e['steps'], 4)
                self.assertEqual(e['backtrack_moves'], 1)
                self.assertEqual(e['actions'][1]['kind'], 'revisit')
                self.assertTrue(e['success'])
                limited = navigate_search(graph, Task('A', 'T'), 3, algorithm)
                self.assertEqual(limited['path'], ['A', 'B', 'A', 'C'])
                self.assertFalse(limited['success'])

    def test_bfs_visits_siblings_before_dfs_deeper_branch(self):
        graph = WikiGraph(set('ABCDT'), {
            'A': ['B', 'C'], 'B': ['A', 'D'], 'C': ['T'], 'D': [], 'T': [],
        })
        bfs = navigate_search(graph, Task('A', 'T'), 12, 'bfs')
        dfs = navigate_search(graph, Task('A', 'T'), 12, 'dfs')
        self.assertEqual(bfs['path'], ['A', 'B', 'A', 'C', 'T'])
        self.assertTrue(bfs['success'])
        self.assertEqual(dfs['path'], ['A', 'B', 'D'])
        self.assertFalse(dfs['success'])
        self.assertEqual(dfs['termination_reason'], 'dead_end')

    def test_no_frontier_teleports_or_unrevealed_adjacency_reads(self):
        class RecordingLinks(dict):
            def get(self, node, default=None):
                reads.append(node)
                return super().get(node, default)

        for algorithm in ('bfs', 'dfs'):
            reads = []
            graph = WikiGraph(set('ABCT'), RecordingLinks({
                'A': ['B', 'C'], 'B': [], 'C': ['T'], 'T': [],
            }))
            e = navigate_search(graph, Task('A', 'T'), 12, algorithm)
            self.assertEqual(e['path'], ['A', 'B'])
            self.assertEqual(reads, ['A', 'B'])
            self.assertFalse(e['success'])
            self.assertEqual(e['termination_reason'], 'dead_end')

    def test_target_visible_on_current_page_is_taken_immediately(self):
        graph = WikiGraph(set('ABCT'), {'A': ['B', 'C', 'T'], 'B': [], 'C': [], 'T': []})
        for algorithm in ('bfs', 'dfs'):
            e = navigate_search(graph, Task('A', 'T'), 1, algorithm)
            self.assertEqual(e['path'], ['A', 'T'])
            self.assertTrue(e['success'])
            e = navigate_search(graph, Task('T', 'T'), 1, algorithm)
            self.assertEqual(e['steps'], 0)
            self.assertTrue(e['success'])

    def test_navigation_metrics_include_failures_and_legal_move_counts(self):
        graph = WikiGraph(set('ABCDT'), {
            'A': ['B', 'C', 'B'], 'B': ['A', 'D'], 'C': ['T'], 'D': [], 'T': [],
        })
        for algorithm in ('bfs', 'dfs'):
            m = time_navigation(graph, [Task('A', 'T'), Task('T', 'A')], 12, algorithm)
            self.assertEqual(m['episode_count'], 2)
            self.assertEqual(m['success_count'], int(algorithm == 'bfs'))
            self.assertEqual(m['success_rate'], .5 if algorithm == 'bfs' else 0)
            self.assertGreaterEqual(m['wall_seconds'], m['total_query_wall_seconds'])
            for e in m['episodes']:
                self.assertEqual(e['steps'], len(e['path']) - 1)
                self.assertLessEqual(e['steps'], 12)
                self.assertTrue(all(b in graph.links.get(a, [])
                                    for a, b in zip(e['path'], e['path'][1:])))

    def test_online_cli_modes_and_move_budget_override(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config, _, _ = self.fixture(root)
            for algorithm in ('bfs', 'dfs'):
                stdout = io.StringIO()
                with redirect_stdout(stdout), \
                        patch('scripts.benchmark_bfs.shortest_path', side_effect=AssertionError('full graph search')):
                    self.assertEqual(main(['--config', str(config), '--splits', 'dev',
                                           '--algorithm', algorithm, '--max-steps', '1']), 0)
                report = json.loads(stdout.getvalue())
                self.assertEqual(report['algorithm'], 'online_' + algorithm + '_navigation')
                self.assertEqual(report['splits']['dev']['max_steps'], 1)
                self.assertEqual(report['splits']['dev']['success_count'], 1)

    def test_queries_are_independent_and_include_failures(self):
        graph = WikiGraph(set('ABCD'), {'A': ['B'], 'B': ['A', 'C'], 'C': [], 'D': []})
        tasks = [Task('A', 'C'), Task('C', 'A'), Task('B', 'B'), Task('A', 'B')]
        # Outer wall clock includes bookkeeping; per-query clocks measure only BFS.
        ticks = [0, 1, 3, 4, 7, 8, 12, 13, 18, 20]
        with patch('scripts.benchmark_bfs.time.perf_counter_ns', side_effect=ticks), \
                patch('scripts.benchmark_bfs.shortest_path', wraps=shortest_path) as bfs:
            report = time_bfs(graph, tasks, max_steps=1)
        self.assertEqual(bfs.call_count, len(tasks))
        self.assertEqual([(c.args[1], c.args[2]) for c in bfs.call_args_list],
                         [(t.start, t.target) for t in tasks])
        self.assertEqual(report['episode_count'], 4)
        self.assertEqual(report['success_count'], 2)
        self.assertEqual(report['unreachable_count'], 1)
        self.assertEqual([e['path'] for e in report['episodes']],
                         [['A', 'B', 'C'], None, ['B'], ['A', 'B']])
        self.assertAlmostEqual(report['wall_seconds'], 20e-9)
        self.assertAlmostEqual(report['total_query_wall_seconds'], 14e-9)
        self.assertAlmostEqual(report['mean_wall_seconds_per_task'], 5e-9)
        self.assertAlmostEqual(report['median_query_wall_seconds'], 3.5e-9)
        self.assertAlmostEqual(report['p95_query_wall_seconds'], 5e-9)

    def fixture(self, root):
        data = root / 'data' / 'processed'
        data.mkdir(parents=True)
        (data / 'graph.json').write_text(json.dumps({
            'articles': ['A', 'B', 'C'], 'links': {'A': ['B'], 'B': ['C'], 'C': []},
        }))
        tasks = root / 'tasks.json'
        tasks.write_text(json.dumps([asdict(Task('A', 'C')), asdict(Task('B', 'C'))]))
        manifest = root / 'manifest.json'
        manifest.write_text(json.dumps({
            'output_sha256': {str(tasks): sha256_file(tasks)},
            'pair_assignments': [{'start': 'A', 'target': 'C', 'split': 'dev'},
                                 {'start': 'B', 'target': 'C', 'split': 'dev'}],
        }))
        config = root / 'remote.toml'
        config.write_text(f'data_dir = "{data.parent}"\ndev_tasks = "{tasks}"\n'
                          f'split_manifest = "{manifest}"\n')
        return config, tasks, manifest

    def test_frozen_selection_subset_and_no_input_changes_or_api_calls(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config, tasks, manifest = self.fixture(root)
            originals = {p: p.read_bytes() for p in root.rglob('*') if p.is_file()}
            output = root / 'bfs.json'
            with redirect_stdout(io.StringIO()), \
                    patch('wiki_speedrun.remote.make_client', side_effect=AssertionError('API call')):
                self.assertEqual(main(['--config', str(config), '--splits', 'dev',
                                       '--max-tasks', '1', '--output', str(output)]), 0)
            report = json.loads(output.read_text())
            dev = report['splits']['dev']
            self.assertEqual(dev['episode_count'], 1)
            self.assertEqual(dev['selection']['split_count'], 2)
            self.assertTrue(dev['selection']['subset'])
            self.assertEqual(dev['selection']['tasks_file_sha256'], sha256_file(tasks))
            self.assertEqual(dev['ordered_tasks_sha256'], object_fingerprint([asdict(Task('A', 'C'))]))
            self.assertTrue(all(p.read_bytes() == value for p, value in originals.items()))

    def test_training_config_selects_its_frozen_data(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            _, tasks, manifest = self.fixture(root)
            config = root / 'training.toml'
            config.write_text(f'data_dir = "{root / "data"}"\n[data]\n'
                              f'dev_tasks = "{tasks}"\nsplit_manifest = "{manifest}"\n'
                              '[rl]\nmax_steps = 1\n')
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(['--training-config', str(config), '--splits', 'dev']), 0)
            dev = json.loads(stdout.getvalue())['splits']['dev']
            self.assertEqual(dev['episode_count'], 2)
            self.assertEqual(dev['success_count'], 1)
            self.assertFalse(dev['selection']['subset'])

    def test_manifest_mismatch_fails_before_search_and_output(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config, tasks, _ = self.fixture(root)
            tasks.write_text(json.dumps([asdict(Task('B', 'C'))]))
            output = root / 'bfs.json'
            with redirect_stderr(io.StringIO()), patch('scripts.benchmark_bfs.time_bfs') as bfs:
                self.assertEqual(main(['--config', str(config), '--splits', 'dev',
                                       '--output', str(output)]), 1)
            bfs.assert_not_called()
            self.assertFalse(output.exists())

    def test_custom_tasks_and_existing_output_protection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config, tasks, manifest = self.fixture(root)
            manifest.unlink()  # Explicit custom tasks do not require a frozen manifest.
            output = root / 'bfs.json'
            command = ['--config', str(config), '--tasks', str(tasks), '--output', str(output)]
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(command), 0)
            original = output.read_bytes()
            self.assertEqual(json.loads(original)['splits']['custom']['episode_count'], 2)
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(command), 1)
            self.assertEqual(output.read_bytes(), original)

    def test_invalid_limits_and_missing_articles(self):
        graph = WikiGraph({'A'}, {'A': []})
        for tasks, limit in [([], 1), ([Task('A', 'A')], 0), ([Task('A', 'missing')], 1)]:
            with self.subTest(tasks=tasks, limit=limit), self.assertRaises(ValueError):
                time_bfs(graph, tasks, limit)
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--max-tasks', '0']), 1)
            self.assertEqual(main(['--max-steps', '0']), 1)
        for algorithm in ('bfs', 'dfs'):
            with self.assertRaises(ValueError):
                time_navigation(graph, [], 12, algorithm)
            with self.assertRaises(ValueError):
                navigate_search(graph, Task('A', 'missing'), 12, algorithm)


if __name__ == '__main__':
    unittest.main()
