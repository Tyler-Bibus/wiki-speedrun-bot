"""Parallel episode completion, exact-once dispatch, limits, and resume."""
from dataclasses import replace
import fcntl
import io
import json
from pathlib import Path
import tempfile
from threading import Barrier, Event, Lock
import time
import unittest
from unittest.mock import patch

from wiki_speedrun.comparison import validate_run
from wiki_speedrun.data.load import WikiGraph
from wiki_speedrun.data.tasks import Task
from wiki_speedrun.models.openrouter import OpenRouterChatClient, OpenRouterDecisionClient, RemoteError
from wiki_speedrun.remote import read_rows, run_remote
from wiki_speedrun.remote_config import RemoteConfig
from wiki_speedrun.remote_parallel import completed_episodes, run_remote_parallel
from wiki_speedrun.remote_policy import LLMPolicy, RemoteDecisionPolicy


def reply():
    return io.BytesIO(json.dumps({
        'id': 'fixture', 'model': 'snapshot', 'provider': 'fixture', 'usage': {'cost': .01},
        'choices': [{'message': {'content': '{"next_article":"T"}'}, 'finish_reason': 'stop'}],
    }).encode())


class ParallelEpisodeTests(unittest.TestCase):
    def fixture(self, root, count=8, **limits):
        starts = [f'A{i}' for i in range(count)]
        graph = WikiGraph(set(starts) | {'T', 'X'}, {s: ['T', 'X'] for s in starts} | {'T': [], 'X': []})
        tasks = [Task(s, 'T') for s in starts]
        config = RemoteConfig(protocol='llm_full_list_json', model='fixture-model', run_dir=str(root), **limits)
        return config, graph, tasks, {'split': 'dev', 'split_count': count, 'max_tasks': None, 'subset': False}

    def factory(self, config, transport):
        def create():
            client = OpenRouterChatClient('fixture-key', config.model, retries=0, transport=transport)
            return client, LLMPolicy(client, config), None
        return create

    def test_out_of_order_successes_survive_failure_and_resume_without_duplicates(self):
        barrier = Barrier(2)
        lock = Lock()
        fail = {'A0'}
        calls = []
        def transport(request, timeout):
            source = json.loads(json.loads(request.data)['messages'][1]['content'])['current_article']
            with lock:
                calls.append(source)
                first = len(calls) <= 2
            if first:
                barrier.wait(timeout=5)
            if source in fail:
                raise RemoteError('provider_unavailable', 'fixture outage')
            return reply()
        with tempfile.TemporaryDirectory() as td:
            config, graph, tasks, selection = self.fixture(td)
            factory = self.factory(config, transport)
            first = run_remote_parallel(config, tasks, selection, workers=3, graph=graph, worker_factory=factory)
            folder = Path(first['run_dir'])
            self.assertEqual(first['status'], 'incomplete')
            self.assertEqual(first['completed_tasks'], 7)
            self.assertEqual(first['failed_tasks']['0']['kind'], 'provider_unavailable')
            self.assertEqual(len(calls), 8)
            self.assertEqual(read_rows(folder / 'episodes.jsonl'), [])
            saved, prefix = completed_episodes(folder, tasks)
            self.assertEqual(set(saved), set(range(1, 8)))
            self.assertEqual(prefix, 0)
            original_files = {p.name: p.read_bytes() for p in (folder / 'completed').glob('*.json')}
            fail.clear()
            result = run_remote_parallel(config, tasks, selection, workers=2, graph=graph,
                                         resume=folder, worker_factory=factory)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(calls.count('A0'), 2)
            for source in tasks[1:]:
                self.assertEqual(calls.count(source.start), 1)
            for name, raw in original_files.items():
                self.assertEqual((folder / 'completed' / name).read_bytes(), raw)
            rows = read_rows(folder / 'episodes.jsonl')
            self.assertEqual([e['episode_id'] for e in rows], [f'eval-{i:06d}' for i in range(8)])
            self.assertEqual(result['http_attempts'], 9)
            self.assertAlmostEqual(result['known_cost'], .08)
            validate_run(folder, graph)
            unchanged_calls = list(calls)
            run_remote_parallel(config, tasks, selection, workers=4, graph=graph, resume=folder, worker_factory=factory)
            self.assertEqual(calls, unchanged_calls)

    def test_request_cap_reserves_slots_across_workers(self):
        barrier = Barrier(2)
        calls = []
        lock = Lock()
        def transport(request, timeout):
            with lock:
                calls.append(request)
            barrier.wait(timeout=5)
            return reply()
        with tempfile.TemporaryDirectory() as td:
            config, graph, tasks, selection = self.fixture(td, max_requests=2)
            result = run_remote_parallel(config, tasks, selection, workers=6, graph=graph,
                                         worker_factory=self.factory(config, transport))
            self.assertEqual(len(calls), 2)
            self.assertEqual(result['http_attempts'], 2)
            self.assertEqual(result['completed_tasks'], 2)
            self.assertEqual(result['status'], 'incomplete')
            status = json.loads((Path(result['run_dir']) / 'status.json').read_text())
            self.assertEqual(status['error']['kind'], 'budget')

    def test_exhausted_generation_failure_keeps_pool_running_and_resumes_only_failed_task(self):
        calls, failing = [], {'A0'}
        def transport(request, timeout):
            source = json.loads(json.loads(request.data)['messages'][1]['content'])['current_article']
            calls.append(source)
            if source in failing:
                return io.BytesIO(json.dumps({
                    'id': 'fixture', 'model': 'snapshot', 'provider': 'fixture', 'usage': {'cost': .01},
                    'choices': [{'message': {'content': None}, 'finish_reason': 'error'}],
                }).encode())
            return reply()
        with tempfile.TemporaryDirectory() as td:
            config, graph, tasks, selection = self.fixture(td, count=8)
            def factory():
                client = OpenRouterChatClient('fixture-key', config.model, transport=transport, sleep=lambda _: None)
                return client, LLMPolicy(client, config), None
            first = run_remote_parallel(config, tasks, selection, workers=2, graph=graph, worker_factory=factory)
            self.assertEqual(first['completed_tasks'], 7)
            self.assertEqual(first['failed_tasks']['0']['kind'], 'provider_unavailable')
            self.assertEqual(calls.count('A0'), 3)
            failing.clear()
            result = run_remote_parallel(config, tasks, selection, workers=2, graph=graph,
                                         resume=first['run_dir'], worker_factory=factory)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['http_attempts'], 11)
            self.assertAlmostEqual(result['known_cost'], .11)
            self.assertEqual(calls.count('A0'), 4)
            for task in tasks[1:]:
                self.assertEqual(calls.count(task.start), 1)
            validate_run(first['run_dir'], graph)

    def test_serial_prefix_is_preserved_when_switching_to_parallel(self):
        calls = []
        def transport(request, timeout):
            calls.append(request)
            return reply()
        with tempfile.TemporaryDirectory() as td:
            config, graph, tasks, selection = self.fixture(td, max_requests=1)
            client = OpenRouterChatClient('fixture-key', config.model, retries=0, transport=transport)
            first = run_remote(config, tasks, selection, graph=graph, client=client)
            folder = Path(first['run_dir'])
            original_prefix = (folder / 'episodes.jsonl').read_bytes()
            provenance = (folder / 'provenance.json').read_bytes()
            config = replace(config, max_requests=100)
            result = run_remote_parallel(config, tasks, selection, workers=4, graph=graph, resume=folder,
                                         worker_factory=self.factory(config, transport))
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(calls), len(tasks))
            self.assertTrue((folder / 'episodes.jsonl').read_bytes().startswith(original_prefix))
            self.assertEqual((folder / 'provenance.json').read_bytes(), provenance)
            validate_run(folder, graph)

    def test_global_pacing_is_shared_by_all_workers(self):
        times = []
        def transport(request, timeout):
            times.append(time.monotonic())
            return reply()
        with tempfile.TemporaryDirectory() as td:
            config, graph, tasks, selection = self.fixture(td, count=4, min_request_interval=.02)
            result = run_remote_parallel(config, tasks, selection, workers=4, graph=graph,
                                         worker_factory=self.factory(config, transport))
            self.assertEqual(result['status'], 'complete')
            self.assertTrue(all(b - a >= .015 for a, b in zip(times, times[1:])))
            self.assertGreater(result['throttle_wait_ms'], 0)

    def test_source_changes_and_duplicate_run_owners_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            config, graph, tasks, selection = self.fixture(td, count=1)
            factory = self.factory(config, lambda request, timeout: reply())
            first = run_remote_parallel(config, tasks, selection, workers=1, graph=graph, worker_factory=factory)
            folder = Path(first['run_dir'])
            with (folder / 'run.lock').open('a+') as lease:
                fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(ValueError, 'already owns'):
                    run_remote_parallel(config, tasks, selection, workers=2, graph=graph, resume=folder, worker_factory=factory)
            p = folder / 'provenance.json'
            value = json.loads(p.read_text()); value['source_sha256'] = 'changed'
            p.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                run_remote_parallel(config, tasks, selection, workers=2, graph=graph, resume=folder, worker_factory=factory)

    def test_cancellation_stops_new_requests_and_preserves_finished_episodes(self):
        cancel = Event()
        calls = []
        def transport(request, timeout):
            calls.append(request)
            cancel.set()
            return reply()
        with tempfile.TemporaryDirectory() as td:
            config, graph, tasks, selection = self.fixture(td, count=20)
            result = run_remote_parallel(config, tasks, selection, workers=3, graph=graph,
                                         cancel_event=cancel, worker_factory=self.factory(config, transport))
            self.assertLessEqual(len(calls), 3)
            self.assertEqual(result['completed_tasks'], len(calls))
            self.assertEqual(result['status'], 'incomplete')

    def test_durable_results_recover_a_missing_ordered_log_without_requests(self):
        calls = []
        def transport(request, timeout):
            calls.append(request)
            return reply()
        with tempfile.TemporaryDirectory() as td:
            config, graph, tasks, selection = self.fixture(td, count=4)
            factory = self.factory(config, transport)
            first = run_remote_parallel(config, tasks, selection, workers=3, graph=graph, worker_factory=factory)
            folder = Path(first['run_dir'])
            original = (folder / 'episodes.jsonl').read_bytes()
            (folder / 'episodes.jsonl').unlink()
            result = run_remote_parallel(config, tasks, selection, workers=2, graph=graph, resume=folder, worker_factory=factory)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(len(calls), 4)
            self.assertEqual(read_rows(folder / 'episodes.jsonl'), [json.loads(row) for row in original.splitlines()])
            validate_run(folder, graph)

    def test_decision_tournament_trace_and_session_ids_remain_valid(self):
        class Budgeter:
            metadata = {'fixture': True}
            def __init__(self, *args): pass
            def fits(self, request): return True
            def token_counts(self, request): return {'state': 1, 'row': 2}
            def state_token_count(self, state): return 1
        def transport(request, timeout):
            payload = json.loads(request.data)
            ids = list(payload['questions']['next_link']['criteria'])
            return io.BytesIO(json.dumps({
                'id': 'fixture', 'model': 'snapshot', 'provider': 'fixture', 'usage': {'cost': .01},
                'answers': {'next_link': {'type': 'choice', 'choice': ids[0],
                                          'probabilities': {key: 1 / len(ids) for key in ids}}},
            }).encode())
        with tempfile.TemporaryDirectory() as td:
            config, graph, tasks, selection = self.fixture(td, count=6)
            config = replace(config, protocol='matched_kev_tournament')
            def factory():
                client = OpenRouterDecisionClient('fixture-key', config.model, retries=0, transport=transport)
                return client, RemoteDecisionPolicy(client, Budgeter(), config), Budgeter.metadata
            result = run_remote_parallel(config, tasks, selection, workers=3, graph=graph,
                                         tokenizer_identity=Budgeter.metadata, worker_factory=factory)
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['http_attempts'], 6)
            with patch('wiki_speedrun.models.kev_budget.KevContextBudgeter', Budgeter):
                validate_run(result['run_dir'], graph)


if __name__ == '__main__':
    unittest.main()
