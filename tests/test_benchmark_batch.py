"""Shared spending bounds and full-split background-worker completion gates."""
from argparse import Namespace
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from wiki_speedrun.batch import command, full_coverage, main as batch_main, retry_delay, worker
from wiki_speedrun.benchmark import MODELS
from wiki_speedrun.cli import parser
from wiki_speedrun.models.openrouter import RemoteError
from wiki_speedrun.remote import append
from wiki_speedrun.spending import SuiteBudget


class SuiteBudgetTests(unittest.TestCase):
    def test_cost_is_shared_across_probes_runs_and_restarts(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            for name,cost in [('probes/m','0.10'),('runs/a','0.15')]:
                p=root/name;p.mkdir(parents=True)
                append(p/'requests.jsonl', {'response':{'usage':{'cost':float(cost)}}})
            budget=SuiteBudget(root,.30)
            self.assertAlmostEqual(budget.known_cost,.25);budget.check()
            row={'response':{'usage':{'cost':.08}}}
            append(root/'runs/a/requests.jsonl',row);budget.observe(row)
            with self.assertRaises(RemoteError):budget.check()
            restored=SuiteBudget(root,.30)
            self.assertAlmostEqual(restored.known_cost,.33)
            with self.assertRaises(RemoteError):restored.check()
            self.assertEqual(json.loads((root/'budget.json').read_text())['http_attempts'],3)

    def test_missing_success_cost_stops_but_failed_attempt_is_retained(self):
        with tempfile.TemporaryDirectory() as td:
            b=SuiteBudget(td,9)
            b.observe({'error':{'kind':'rate_limit'}});b.check()
            b.observe({'response':{'usage':None}})
            with self.assertRaisesRegex(RemoteError,'lacks cost'):b.check()
            d=json.loads((Path(td)/'budget.json').read_text())
            self.assertEqual(d['unknown_cost_attempts'],2)
            self.assertEqual(d['unpriced_responses'],1)

    def test_invalid_limits(self):
        for value in (0,-1,float('nan'),float('inf'),True):
            with tempfile.TemporaryDirectory() as td, self.assertRaises(ValueError):
                SuiteBudget(td,value)


class BatchTests(unittest.TestCase):
    def test_default_lineup_and_fresh_output_match_between_entry_points(self):
        args = parser().parse_args(['benchmark'])
        self.assertEqual(args.models, ['minimax', 'deepseek', 'jev'])
        self.assertEqual(set(args.models), set(MODELS))
        self.assertEqual(args.output, Path('runs/openrouter-paid-comparison'))
        with patch('wiki_speedrun.batch.worker', return_value=0) as run:
            self.assertEqual(batch_main([]), 0)
        batch_args = run.call_args.args[0]
        self.assertEqual(batch_args.models, args.models)
        self.assertEqual(batch_args.output, Path('runs/openrouter-paid-full'))
        self.assertEqual(batch_args.report, Path('docs/jev-paid-full-results.md'))
        self.assertEqual(batch_args.max_total_cost, args.max_total_cost)

    def test_retry_classification(self):
        for kind in ('rate_limit','network','provider_unavailable'):
            self.assertEqual(retry_delay({'a':{'kind':kind},'b':{'kind':'setup_dependency'}}),300)
        self.assertEqual(retry_delay({'a':{'kind':'daily_quota'}}),3600)
        self.assertEqual(retry_delay({'a':{'kind':'http','message':'OpenRouter HTTP 503 (http)'}}),300)
        for kind in ('suite_budget','auth','schema','unsupported_parameters','model_changed','runtime'):
            self.assertIsNone(retry_delay({'a':{'kind':kind}}))

    def test_full_command_never_has_a_sample_limit(self):
        args=Namespace(output=Path('runs/full'),max_total_cost=9,models=['minimax','deepseek','jev'],
                       splits=['dev','test'],include_local=True)
        cmd=command(args)
        self.assertNotIn('--max-tasks',cmd)
        self.assertIn('--max-total-cost',cmd)
        self.assertIn('test',cmd)

    def test_full_snap_and_concurrency_options_reach_child(self):
        args=Namespace(output=Path('runs/full'),max_total_cost=9,models=['minimax'],
                       splits=['dev'],include_local=False,remote_workers=2,workers_per_model=8,
                       training_config=Path('configs/snap-full.toml'),
                       sft_checkpoint=Path('checkpoints/new-sft/update-002000'))
        cmd=command(args)
        self.assertEqual(cmd[cmd.index('--remote-workers')+1], '2')
        self.assertEqual(cmd[cmd.index('--workers-per-model')+1], '8')
        self.assertEqual(cmd[cmd.index('--training-config')+1], str(args.training_config))
        self.assertEqual(cmd[cmd.index('--sft-checkpoint')+1], str(args.sft_checkpoint))
        self.assertIn('--no-include-local',cmd)

    def test_pilot_cannot_pass_full_coverage_gate(self):
        self.assertFalse(full_coverage({'settings':{'max_tasks':5}},['minimax'],['dev'],{'dev':179},False))
        self.assertFalse(full_coverage({'settings':{'max_tasks':None},'runs':{}},['minimax'],['dev'],{'dev':179},False))

    def test_worker_retries_then_publishes_only_complete_splits(self):
        with tempfile.TemporaryDirectory() as td:
            out=Path(td)/'run';report=Path(td)/'full.md'
            args=Namespace(output=out,report=report,max_total_cost=9,models=['minimax'],
                           splits=['dev','test'],include_local=False)
            calls=[]
            def run(cmd):
                calls.append(cmd)
                state={'settings':{'max_tasks':None},'runs':{},'errors':{}}
                if len(calls)==1:
                    state['errors']={'minimax-dev':{'kind':'provider_unavailable'}}
                else:
                    for split,count in [('dev',179),('test',175)]:
                        folder=out/split;folder.mkdir()
                        (folder/'metrics.json').write_text(json.dumps({'status':'complete','episode_count':count}))
                        state['runs']['minimax-'+split]=str(folder)
                        (out/(split+'-results.md')).write_text('# Navigation comparison\n\n'+str(count)+' full tasks\n')
                    (out/'budget.json').write_text(json.dumps({'known_cost':1.2,'unknown_cost_attempts':0,'max_total_cost':9}))
                (out/'benchmark.json').write_text(json.dumps(state))
                return SimpleNamespace(returncode=1 if len(calls)==1 else 0)
            with patch('wiki_speedrun.batch.subprocess.run',side_effect=run), patch('wiki_speedrun.batch.retry_delay',return_value=0):
                self.assertEqual(worker(args),0)
            self.assertEqual(len(calls),2)
            job=json.loads((out/'job.json').read_text())
            self.assertEqual(job['status'],'complete')
            self.assertEqual(job['split_task_counts'],{'dev':179,'test':175})
            self.assertTrue(report.exists())
            self.assertIn('$1.200000',report.read_text())


if __name__=='__main__':
    unittest.main()
