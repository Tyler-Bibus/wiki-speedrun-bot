"""Offline API, navigation, persistence, and paired-evidence contract tests."""
import copy
from dataclasses import asdict, replace
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from wiki_speedrun.choices import TournamentPolicy, make_request
from wiki_speedrun.credentials import load_api_key
from wiki_speedrun.data.load import WikiGraph
from wiki_speedrun.data.tasks import Task
from wiki_speedrun.environment import NavigationEnvironment
from wiki_speedrun.models.openrouter import (OpenRouterChatClient, OpenRouterDecisionClient,
                                             NoRedirect, RemoteError, strict_json)
from wiki_speedrun.remote import run_remote, selected_tasks, read_rows, recover_tail
from wiki_speedrun.remote_config import RemoteConfig
from wiki_speedrun.remote_policy import LLMPolicy, RemoteDecisionPolicy
from wiki_speedrun.comparison import validate_run, write_comparison


def decision(ids=('link_0', 'link_1'), selected=None, **extra):
    return {'id': 'req-1', 'model': 'snapshot', 'provider': 'fixture',
            'answers': {'next_link': {'type': 'choice', 'choice': selected or ids[0],
                                     'probabilities': {k: 1 / len(ids) for k in ids}}}, **extra}


def chat(content='{"next_article":"B"}', finish='stop', **extra):
    return {'id': 'req-1', 'model': 'snapshot', 'provider': 'fixture',
            'choices': [{'message': {'content': content}, 'finish_reason': finish}], **extra}


class Transport:
    def __init__(self, values):
        self.values = iter(values)
        self.calls = []

    def __call__(self, request, timeout):
        self.calls.append((request, timeout))
        value = next(self.values)
        if isinstance(value, Exception):
            raise value
        if callable(value):
            value = value(json.loads(request.data))
        return io.BytesIO(json.dumps(value).encode())


def graph():
    return WikiGraph({'A', 'B', 'C', 'D'}, {'A': ['B', 'C'], 'B': ['C', 'D'], 'C': [], 'D': []})


class CredentialsTests(unittest.TestCase):
    def test_env_precedence_and_limited_file_format(self):
        with tempfile.TemporaryDirectory() as td, patch.dict(os.environ, {}, clear=True):
            p = Path(td) / 'key'
            for raw in ('OPENROUTER_API_KEY=abc', ' # comment\n\nOPENROUTER_API_KEY="abc"', "OPENROUTER_API_KEY='abc'"):
                p.write_text(raw)
                self.assertEqual(load_api_key(p), 'abc')
            with patch.dict(os.environ, {'OPENROUTER_API_KEY': 'env-key'}):
                self.assertEqual(load_api_key('/nonexistent'), 'env-key')
            for raw in ('', 'OTHER=secret', 'OPENROUTER_API_KEY=', 'OPENROUTER_API_KEY="secret',
                        'OPENROUTER_API_KEY=$(bad)', 'OPENROUTER_API_KEY=secret\nOTHER=2'):
                p.write_text(raw)
                with self.subTest(raw=raw), self.assertRaises(ValueError) as ctx:
                    load_api_key(p)
                self.assertNotIn('secret', str(ctx.exception))


class TransportTests(unittest.TestCase):
    def request(self):
        return make_request({'current_title': 'A'}, ['link_0', 'link_1'], ['B', 'C'])

    def test_fixed_origin_auth_copy_and_unknown_usage(self):
        t = Transport([decision()]); records = []
        c = OpenRouterDecisionClient('secret', 'requested', transport=t, on_attempt=records.append)
        request = self.request(); before = copy.deepcopy(request)
        result = c.score(request)
        self.assertEqual(request, before)
        actual, timeout = t.calls[0]
        self.assertEqual(actual.full_url, 'https://openrouter.ai/api/alpha/decisions')
        self.assertEqual(actual.get_header('Authorization'), 'Bearer secret')
        self.assertEqual(json.loads(actual.data)['model'], 'requested')
        self.assertIsNone(result['usage'])
        self.assertNotIn('secret', json.dumps(records))

    def test_redacts_provider_echo(self):
        t = Transport([decision(usage={'cost': 0}, note='secret')]); records = []
        OpenRouterDecisionClient('secret', 'x', transport=t, on_attempt=records.append).score(self.request())
        self.assertNotIn('secret', json.dumps(records))

    def test_validation_rejects_contract_errors(self):
        fixtures = []
        for update in ({'type': 'noul'}, {'choice': 'made_up'}, {'probabilities': {'link_0': 1}},
                       {'probabilities': {'link_0': 0.9, 'link_1': 0.9}}, {'confidence': 2},
                       {'probabilities': {'link_0': -0.1, 'link_1': 1.1}},
                       {'probabilities': {'link_0': True, 'link_1': 0}}):
            d = decision(); d['answers']['next_link'].update(update); fixtures.append(d)
        fixtures += [decision(usage={'cost': -1}), decision(usage={'input_tokens': 1.5}),
                     decision(usage={'completion_tokens_details': {'reasoning_tokens': -1}}),
                     decision(usage={'prompt_tokens_details': 'broken'}), decision(provider=None), {'id': 'x'}]
        for value in fixtures:
            with self.subTest(value=value), self.assertRaises(RemoteError):
                OpenRouterDecisionClient('key', 'm', transport=Transport([value])).score(self.request())

    def test_retry_and_sanitized_http(self):
        for status in (429, 500, 502, 503, 504, 529):
            e = urllib.error.HTTPError('https://secret', status, 'secret', {'Retry-After': '999'}, io.BytesIO(b'secret'))
            t = Transport([e, decision()]); waits = []; records = []
            c = OpenRouterDecisionClient('secret', 'm', transport=t, sleep=waits.append, on_attempt=records.append)
            self.assertEqual(c.score(self.request())['attempts'], 2)
            self.assertEqual(waits, [30])
            self.assertEqual(len(records), 2)
            self.assertNotIn('secret', json.dumps(records))
        for status in (400, 401, 402, 403, 404, 422):
            t = Transport([urllib.error.HTTPError('https://secret', status, 'secret', {}, io.BytesIO(b'secret'))])
            with self.assertRaises(RemoteError) as ctx:
                OpenRouterDecisionClient('secret', 'm', transport=t).score(self.request())
            self.assertNotIn('secret', str(ctx.exception)); self.assertEqual(len(t.calls), 1)

    def test_decision_distribution_accepts_accumulated_rounding_without_changing_choice(self):
        ids = tuple(f'link_{i}' for i in range(58))
        request = make_request({'current_title': 'A'}, list(ids), list(ids))
        probs = dict.fromkeys(ids, 0)
        probs.update(link_0=0.23, link_1=0.14, link_2=0.14, link_3=0.08, link_4=0.40)
        for total in (0.99, 1.01):
            probs['link_4'] = 0.40 if total == 0.99 else 0.42
            response = decision(ids, selected='link_4')
            response['answers']['next_link']['probabilities'] = dict(probs)
            client = OpenRouterDecisionClient('key', 'm', transport=Transport([response]))
            result = client.score(request)
            self.assertEqual(result['choice'], 'link_4')
            self.assertEqual(result['probabilities'], probs)
            self.assertAlmostEqual(sum(result['probabilities'].values()), total)
        for invalid in (dict.fromkeys(ids, 0), dict.fromkeys(ids, 0.10), {'link_0': 1}):
            response['answers']['next_link']['probabilities'] = invalid
            with self.assertRaises(RemoteError):
                OpenRouterDecisionClient('key', 'm', transport=Transport([response])).score(request)

    def test_embedded_provider_errors_retry_and_preserve_usage(self):
        for status in (429, 500, 502, 503, 504, 529):
            failed = chat(finish='error', usage={'cost': .01})
            failed['choices'][0]['error'] = {'code': status, 'message': 'secret provider details'}
            t = Transport([failed, chat(usage={'cost': .02})]); records = []; waits = []
            client = OpenRouterChatClient('secret', 'm', transport=t, on_attempt=records.append, sleep=waits.append)
            result = client.complete({})
            self.assertEqual(result['attempts'], 2)
            self.assertEqual(waits, [30 if status == 429 else 1])
            self.assertEqual(records[0]['response']['usage']['cost'], .01)
            self.assertEqual(records[1]['response']['usage']['cost'], .02)
            self.assertNotIn('secret', json.dumps(records))
        for code in (400, 401, None):
            failed['choices'][0]['error']['code'] = code
            t = Transport([failed])
            with self.assertRaises(RemoteError) as ctx:
                OpenRouterChatClient('key', 'm', transport=t).complete({})
            self.assertEqual(ctx.exception.kind, 'api')
            self.assertEqual(len(t.calls), 1)

    def test_embedded_provider_error_exhaustion_remains_retryable_by_worker(self):
        failed = chat(finish='error', usage={'cost': 0})
        failed['choices'][0]['error'] = {'code': 502}
        t = Transport([failed, failed, failed]); records = []
        client = OpenRouterChatClient('key', 'm', transport=t, on_attempt=records.append, sleep=lambda _: None)
        with self.assertRaises(RemoteError) as ctx:
            client.complete({})
        self.assertEqual(ctx.exception.kind, 'provider_unavailable')
        self.assertEqual(len(t.calls), 3)
        self.assertEqual(len(records), 3)
        self.assertEqual(records[-1]['error']['kind'], 'provider_unavailable')

    def test_generation_failure_without_error_details_retries_and_preserves_cost(self):
        failed = chat(content=None, finish='error', usage={'cost': .01})
        failed['choices'][0]['native_finish_reason'] = None
        failed['choices'][0]['message']['reasoning'] = 'partial generation'
        records, waits = [], []
        transport = Transport([failed, chat(usage={'cost': .02})])
        client = OpenRouterChatClient('key', 'm', transport=transport,
                                      on_attempt=records.append, sleep=waits.append)
        self.assertEqual(client.complete({})['attempts'], 2)
        self.assertEqual(waits, [1])
        self.assertEqual(records[0]['error']['kind'], 'provider_unavailable')
        self.assertAlmostEqual(sum(r['response']['usage']['cost'] for r in records), .03)
        transport = Transport([failed] * 3)
        client = OpenRouterChatClient('key', 'm', transport=transport, sleep=lambda _: None)
        with self.assertRaises(RemoteError) as ctx:
            client.complete({})
        self.assertEqual(ctx.exception.kind, 'provider_unavailable')
        self.assertEqual(len(transport.calls), 3)

    def test_generation_failure_with_explicit_nontransient_error_is_not_retried(self):
        for container in ('choice', 'message'):
            failed = chat(content=None, finish='error', usage={'cost': 0})
            target = failed['choices'][0]
            if container == 'message':
                target = target['message']
            target['error'] = {'code': 401}
            transport = Transport([failed])
            client = OpenRouterChatClient('key', 'm', transport=transport)
            with self.assertRaises(RemoteError) as ctx:
                client.complete({})
            self.assertEqual(ctx.exception.kind, 'api')
            self.assertEqual(len(transport.calls), 1)

    def test_top_level_provider_error_envelope_retries(self):
        t = Transport([{'error': {'code': '503', 'message': 'private upstream detail'}}, chat()])
        client = OpenRouterChatClient('key', 'm', transport=t, sleep=lambda _: None)
        self.assertEqual(client.complete({})['attempts'], 2)

    def test_exhaustion_and_ambiguous_timeout(self):
        t = Transport([urllib.error.HTTPError('x', 429, 'limited', {}, io.BytesIO()) for _ in range(3)])
        with self.assertRaises(RemoteError):
            OpenRouterDecisionClient('k', 'm', transport=t, sleep=lambda _: None).score(self.request())
        self.assertEqual(len(t.calls), 3)
        t = Transport([TimeoutError('secret')])
        with self.assertRaises(RemoteError):
            OpenRouterDecisionClient('secret', 'm', transport=t).score(self.request())
        self.assertEqual(len(t.calls), 1)

    def test_snapshot_change_and_redirect(self):
        d = decision(); changed = decision(); changed['model'] = 'new'
        c = OpenRouterDecisionClient('key', 'm', transport=Transport([d, changed]))
        c.score(self.request())
        with self.assertRaisesRegex(RemoteError, 'changed'):
            c.score(self.request())
        self.assertIsNone(NoRedirect().redirect_request(None, None, 302, '', {}, 'https://evil'))

    def test_context_limit_and_chat_envelope(self):
        t = Transport([urllib.error.HTTPError('x', 400, '', {}, io.BytesIO(b'context length exceeded'))])
        with self.assertRaises(RemoteError) as ctx:
            OpenRouterChatClient('key', 'm', transport=t).complete({})
        self.assertEqual(ctx.exception.kind, 'context_limit')
        for d in ({'choices': []}, chat() | {'choices': [{'message': None, 'finish_reason': 'stop'}]}):
            with self.assertRaises(RemoteError):
                OpenRouterChatClient('key', 'm', transport=Transport([d])).complete({})


class PolicyTests(unittest.TestCase):
    def config(self, **kwargs):
        return RemoteConfig(protocol='llm_full_list_json', **kwargs)

    def test_full_list_over_255_preserves_duplicates_and_no_budgeter(self):
        g = graph(); g.links['A'] = ['B'] * 300 + ['C']
        c = OpenRouterChatClient('key', 'm', transport=Transport([chat()]))
        p = LLMPolicy(c, self.config())
        result = p.choose(NavigationEnvironment(g, 'A', 'C'))
        self.assertEqual(result['article'], 'B')
        payload = json.loads(c._transport.calls[0][0].data)
        state = json.loads(payload['messages'][1]['content'])
        self.assertEqual(len(state['outgoing_articles']), 301)
        self.assertEqual(payload['provider']['require_parameters'], True)
        self.assertTrue(payload['response_format']['json_schema']['strict'])
        self.assertNotIn('enum', json.dumps(payload['response_format']))
        self.assertEqual(state['history'], ['A'])

    def test_model_authored_invalid_answers_no_retry(self):
        fixtures = [('not json', 'invalid_response'), ('```json\n{}\n```', 'invalid_response'),
                    ('{"next_article":"B","next_article":"C"}', 'invalid_response'),
                    ('{"next_article":"B","extra":1}', 'invalid_response'),
                    ('{"next_article":1}', 'invalid_response'), ('{}', 'invalid_response'),
                    ('{"next_article":"D"}', 'invalid_link'), ('{"next_article":"New"}', 'invalid_page'),
                    ('{"next_article":"b"}', 'invalid_page'), (None, 'invalid_response')]
        for raw, expected in fixtures:
            c = OpenRouterChatClient('key', 'm', transport=Transport([chat(raw)]))
            env = NavigationEnvironment(graph(), 'A', 'D')
            result = LLMPolicy(c, self.config()).choose(env)
            self.assertEqual(result['rejected']['failure'], expected)
            self.assertEqual(env.history, ['A']); self.assertEqual(len(c._transport.calls), 1)
        refused = chat(); refused['choices'][0]['message']['refusal'] = 'Cannot answer'
        c = OpenRouterChatClient('key', 'm', transport=Transport([refused]))
        self.assertEqual(LLMPolicy(c, self.config()).choose(NavigationEnvironment(graph(), 'A', 'D'))['rejected']['failure'], 'invalid_response')
        for finish in ('length', 'content_filter'):
            c = OpenRouterChatClient('key', 'm', transport=Transport([chat(finish=finish)]))
            self.assertEqual(LLMPolicy(c, self.config()).choose(NavigationEnvironment(graph(), 'A', 'D'))['rejected']['failure'], 'invalid_response')

    def test_tournament_all_edges_and_singletons(self):
        class Budget:
            def fits(self, request): return True
            def token_counts(self, request): return {'state': 1, 'row': 2}
            def state_token_count(self, state): return 1
        g = graph(); g.links['A'] = ['B'] * 300 + ['C']
        seen = []
        def respond(payload):
            ids = list(payload['questions']['next_link']['criteria']); seen.append(ids)
            return decision(ids)
        c = OpenRouterDecisionClient('key', 'm', transport=Transport([respond] * 20))
        p = RemoteDecisionPolicy(c, Budget(), RemoteConfig())
        result = p.choose(NavigationEnvironment(g, 'A', 'D'))
        first_round = [n for n in result['nodes'] if n['round'] == 0]
        self.assertEqual([i for n in first_round for i in n['candidate_ids']], [f'link_{i}' for i in range(301)])
        count = len(seen);g.links['A'] = ['B']
        self.assertEqual(p.choose(NavigationEnvironment(g, 'A', 'D'))['nodes'], [])
        self.assertEqual(len(seen), count)


class PersistenceTests(unittest.TestCase):
    def run_fixture(self, root, values, tasks=None, **settings):
        config = RemoteConfig(protocol='llm_full_list_json', run_dir=str(root), **settings)
        client = OpenRouterChatClient('private-key', 'model', transport=Transport(values))
        tasks = tasks or [Task('A', 'C')]
        metrics = run_remote(config, tasks, {'split_count': len(tasks), 'subset': False},
                             graph=graph(), client=client)
        return config, metrics, Path(metrics['run_dir'])

    def test_invalid_episode_continues_denominator_and_prefix(self):
        with tempfile.TemporaryDirectory() as td:
            config, m, folder = self.run_fixture(td, [chat(), chat('{"next_article":"Made_up"}'), chat('{"next_article":"C"}')],
                                                 [Task('A', 'D'), Task('A', 'C')])
            self.assertEqual(m['status'], 'complete'); self.assertEqual(m['episode_count'], 2)
            eps = read_rows(folder / 'episodes.jsonl')
            self.assertEqual(eps[0]['path'], ['A', 'B']); self.assertEqual(eps[0]['steps'], 1)
            self.assertEqual(eps[0]['termination_reason'], 'invalid_page')
            self.assertEqual(m['success_count'], 1); self.assertEqual(m['http_attempts'], 3)
            self.assertEqual(m['unknown_cost_attempts'], 3)
            validate_run(folder, graph())

    def test_incomplete_and_resume_preserve_paid_attempts(self):
        with tempfile.TemporaryDirectory() as td:
            tasks = [Task('A', 'C'), Task('A', 'D')]
            config, m, folder = self.run_fixture(td, [chat('{"next_article":"C"}', usage={'cost': .1}), chat(usage={'cost': .05}), TimeoutError()], tasks)
            self.assertEqual(m['status'], 'incomplete'); self.assertEqual(m['completed_tasks'], 1)
            self.assertEqual(len(read_rows(folder / 'requests.jsonl')), 3)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                validate_run(folder, graph())
            client = OpenRouterChatClient('private-key', 'model', transport=Transport([chat(), chat('{"next_article":"D"}', usage={'cost': .2})]))
            result = run_remote(config, tasks, {'split_count': 2, 'subset': False}, graph=graph(), client=client, resume=folder)
            self.assertEqual(result['status'], 'complete'); self.assertEqual(result['completed_tasks'], 2)
            self.assertEqual(result['http_attempts'], 5)
            self.assertAlmostEqual(result['known_cost'], .35)
            self.assertEqual(len(read_rows(folder / 'transitions.jsonl')), 4)
            self.assertEqual(len(read_rows(folder / 'episodes.jsonl')), 2)
            self.assertNotIn('private-key', ''.join(p.read_text() for p in folder.iterdir()))
            validate_run(folder, graph())

    def test_budget_resume_mismatch_and_completed_invalid_not_replaced(self):
        with tempfile.TemporaryDirectory() as td:
            config, m, folder = self.run_fixture(td, [chat()], max_requests=1)
            self.assertEqual(m['status'], 'incomplete'); self.assertEqual(m['episode_count'], 0)
            self.assertEqual(len(read_rows(folder / 'transitions.jsonl')), 1)
            client = OpenRouterChatClient('private-key', 'model', transport=Transport([]))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                run_remote(replace(config, temperature=1), [Task('A', 'C')], {'split_count': 1, 'subset': False}, graph=graph(), client=client, resume=folder)
        with tempfile.TemporaryDirectory() as td:
            config, m, folder = self.run_fixture(td, [chat('invalid')])
            client = OpenRouterChatClient('private-key', 'model', transport=Transport([]))
            result = run_remote(config, [Task('A', 'C')], {'split_count': 1, 'subset': False}, graph=graph(), client=client, resume=folder)
            self.assertEqual(result['http_attempts'], 1); self.assertEqual(result['success_count'], 0)

    def test_code_identity_mismatch_still_blocks_resume(self):
        with tempfile.TemporaryDirectory() as td:
            config, metrics, folder = self.run_fixture(td, [chat()], max_requests=1)
            path = folder / 'provenance.json'
            provenance = json.loads(path.read_text())
            provenance['source_sha256'] = 'different-source'
            path.write_text(json.dumps(provenance))
            client = OpenRouterChatClient('key', 'model', transport=Transport([]))
            with self.assertRaisesRegex(ValueError, 'identity mismatch'):
                run_remote(config, [Task('A', 'C')], {'split_count': 1, 'subset': False},
                           graph=graph(), client=client, resume=folder)

    def test_pair_validation_rejects_duplicates_and_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            config, m, folder = self.run_fixture(td, [chat('{"next_article":"C"}', usage={'cost': 0})])
            validate_run(folder, graph())
            ep = read_rows(folder / 'episodes.jsonl')[0]
            with (folder / 'episodes.jsonl').open('a') as stream: stream.write(json.dumps(ep) + '\n')
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                validate_run(folder, graph())

    def test_crash_torn_tail_retained(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / 'requests.jsonl';p.write_bytes(b'{"ok":true}\n{"partial"')
            recover_tail(p)
            self.assertEqual(read_rows(p), [{'ok': True}]);self.assertTrue(p.with_suffix('.jsonl.torn').exists())



class AdditionalAcceptanceTests(unittest.TestCase):
    def test_llm_process_does_not_import_gpu_or_tokenizer(self):
        code = ('import sys; from wiki_speedrun.remote import make_policy; '
                'from wiki_speedrun.remote_config import RemoteConfig; '
                'make_policy(RemoteConfig(protocol="llm_full_list_json"),object()); '
                'assert all(k not in sys.modules for k in ("torch","transformers","tokenizers","kev.model"))')
        subprocess.run([sys.executable, '-c', code], check=True)

    def test_budget_cost_stop_before_next_call(self):
        with tempfile.TemporaryDirectory() as td:
            c = RemoteConfig(protocol='llm_full_list_json', run_dir=td, max_cost=.01)
            transport = Transport([chat(usage={'cost': .02})])
            client = OpenRouterChatClient('key', 'model', transport=transport)
            m = run_remote(c, [Task('A', 'D')], {'split_count': 1}, graph=graph(), client=client)
            self.assertEqual(m['status'], 'incomplete');self.assertEqual(m['http_attempts'], 1)
            self.assertEqual(m['known_cost'], .02);self.assertEqual(len(transport.calls), 1)

    def test_zero_inference_terminal_states(self):
        with tempfile.TemporaryDirectory() as td:
            c = RemoteConfig(protocol='llm_full_list_json', run_dir=td)
            g = graph();g.links['A'] = ['B'];g.links['B'] = []
            client = OpenRouterChatClient('key', 'model', transport=Transport([]))
            m = run_remote(c, [Task('A', 'B'),Task('C','D'),Task('D','D')], {'split_count': 3}, graph=g, client=client)
            self.assertEqual(m['http_attempts'], 0);self.assertEqual(m['success_count'], 2)
            self.assertEqual(m['episode_count'], 3)

    def test_manifest_selection_and_subset_validation(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);p=root/'dev.json';p.write_text(json.dumps([{'start':'A','target':'B'}]))
            from wiki_speedrun.training.checkpoints import sha256_file
            manifest=root/'manifest.json';manifest.write_text(json.dumps({
                'output_sha256': {str(p):sha256_file(p)},
                'pair_assignments':[{'start':'A','target':'B','split':'dev'}]}))
            c=RemoteConfig(dev_tasks=str(p),split_manifest=str(manifest))
            tasks,selection=selected_tasks(c,'dev',max_tasks=10)
            self.assertTrue(selection['subset']);self.assertEqual(len(tasks),1)
            with self.assertRaises(ValueError):selected_tasks(c,'dev',p)
            p.write_text('[]')
            with self.assertRaises(ValueError):selected_tasks(c,'dev')

    def test_comparison_requires_equal_protocols_unless_explicit(self):
        with tempfile.TemporaryDirectory() as td:
            c=RemoteConfig(protocol='llm_full_list_json',run_dir=td)
            config2=replace(c,protocol='llm_full_list_json_object',structured_output='json_object')
            folders=[]
            for cfg in (c,config2):
                client=OpenRouterChatClient('key','m',transport=Transport([chat('{"next_article":"C"}')]))
                m=run_remote(cfg,[Task('A','C')],{'split_count':1},graph=graph(),client=client)
                folders.append(m['run_dir'])
            with self.assertRaisesRegex(ValueError,'Protocol mismatch'):
                write_comparison(folders,Path(td)/'result.md',graph=graph())
            write_comparison(folders,Path(td)/'result.md',graph=graph(),allow_system_protocols=True)
            self.assertTrue((Path(td)/'result.json').is_file())
            p=Path(folders[0])/'metrics.json';m=json.loads(p.read_text());m['known_cost']=10;p.write_text(json.dumps(m))
            with self.assertRaisesRegex(ValueError,'usage total'):
                validate_run(folders[0],graph())


class BenchmarkCommandTests(unittest.TestCase):
    def test_one_command_report_and_second_invocation_reuses_completed_work(self):
        from argparse import Namespace
        from wiki_speedrun.benchmark import benchmark_command
        with tempfile.TemporaryDirectory() as td:
            args=Namespace(output=Path(td),max_tasks=1,models=['deepseek'],splits=['dev'],
                           include_local=False,credentials_file=None)
            client=OpenRouterChatClient('key','m',transport=Transport([chat('{"next_article":"C"}',usage={'cost':0})]))
            selection={'split':'dev','max_tasks':1,'subset':True,'split_count':1}
            with patch('wiki_speedrun.benchmark.load_graph',return_value=graph()), \
                 patch('wiki_speedrun.benchmark.selected_tasks',return_value=([Task('A','C')],selection)), \
                 patch('wiki_speedrun.benchmark.probe',return_value=client):
                self.assertEqual(benchmark_command(args),0)
                self.assertEqual(benchmark_command(args),0)
            self.assertEqual(len(client._transport.calls),1)
            self.assertTrue((Path(td)/'report.md').is_file())
            self.assertTrue((Path(td)/'dev-results.json').is_file())
            state=json.loads((Path(td)/'benchmark.json').read_text())
            self.assertEqual(len(state['runs']),1)

    def test_shared_budget_blocks_the_next_split_before_network(self):
        from argparse import Namespace
        from wiki_speedrun.benchmark import benchmark_command
        with tempfile.TemporaryDirectory() as td:
            args=Namespace(output=Path(td),max_tasks=None,models=['deepseek'],splits=['dev','test'],
                           include_local=False,credentials_file=None,max_total_cost=.04)
            client=OpenRouterChatClient('key','m',transport=Transport([chat('{"next_article":"C"}',usage={'cost':.05})]))
            def tasks(config,split,max_tasks=None):
                return [Task('A','C')], {'split':split,'max_tasks':None,'subset':False,'split_count':1}
            with patch('wiki_speedrun.benchmark.load_graph',return_value=graph()), \
                 patch('wiki_speedrun.benchmark.selected_tasks',side_effect=tasks), \
                 patch('wiki_speedrun.benchmark.probe',return_value=client):
                self.assertEqual(benchmark_command(args),1)
            self.assertEqual(len(client._transport.calls),1)
            state=json.loads((Path(td)/'benchmark.json').read_text())
            self.assertEqual(state['errors']['deepseek-test']['kind'],'suite_budget')
            self.assertAlmostEqual(json.loads((Path(td)/'budget.json').read_text())['known_cost'],.05)

    def test_cli_task_and_split_are_mutually_exclusive(self):
        from wiki_speedrun.cli import parser
        with self.assertRaises(SystemExit):
            parser().parse_args(['evaluate','--policy','llm','--tasks','pairs.json','--split','dev'])

if __name__ == '__main__':
    unittest.main()
