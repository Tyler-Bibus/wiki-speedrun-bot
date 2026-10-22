import unittest
import json
import importlib.util
import os
import random
import tempfile
from collections import Counter
from types import SimpleNamespace
from pathlib import Path

from wiki_speedrun.data.load import WikiGraph
from wiki_speedrun.data.tasks import Task
from wiki_speedrun.training.rewards import episode_return, reward_to_go
from wiki_speedrun.training.rollouts import collect_episode
from wiki_speedrun.training.rl import _development_tasks


def reward_config():
    return SimpleNamespace(
        rl=SimpleNamespace(max_steps=2, gamma=1.0),
        model=SimpleNamespace(device="cpu", autocast_dtype="fp32"),
        reward=SimpleNamespace(success=1.0, failure=-1.0, step_cost=0.02),
    )


class FixedPolicy:
    def choose(self, env):
        article = "Target" if env.current == "Start" else "Target"
        node = {
            "round": 0, "node": 0, "request": {},
            "candidate_ids": ["link_0", "link_1"],
            "articles": [article, "Other"],
            "selected_link_id": "link_0", "old_log_prob": -0.5, "entropy": 0.6,
        }
        return {"article": article, "selected_link_id": "link_0", "nodes": [node]}


class TrainingLoopTests(unittest.TestCase):
    def test_development_subset_is_fixed_and_does_not_consume_training_rng(self):
        from wiki_speedrun.training.config import DataConfig, TrainingConfig

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "dev.json"
            tasks = [{"start": f"Start_{index}", "target": f"Target_{index}"}
                     for index in range(10)]
            path.write_text(json.dumps(tasks), encoding="utf-8")
            config = TrainingConfig(data_dir=temp, data=DataConfig(dev_tasks=str(path)))
            config.save.evaluation_max_tasks = 3
            random.seed(71)
            before = random.getstate()
            first = _development_tasks(config)
            after = random.getstate()
            second = _development_tasks(config)
            self.assertEqual(before, after)
            self.assertEqual(first, second)
            self.assertEqual(len(first), 3)

    def test_reward_arithmetic_and_click_returns(self):
        config = reward_config()
        success = {"steps": 2, "termination_reason": "success"}
        failure = {"steps": 2, "termination_reason": "step_limit"}
        self.assertAlmostEqual(episode_return(success, config), 0.96)
        self.assertAlmostEqual(episode_return(failure, config), -1.04)
        self.assertEqual(reward_to_go([-0.02, 0.98]), [0.96, 0.98])
        self.assertAlmostEqual(episode_return({"steps": 0, "termination_reason": "dead_end"}, config), -1.0)

    def test_collection_rewards_only_real_clicks_and_keeps_all_nodes(self):
        graph = WikiGraph(
            {"Start", "Target", "Other"},
            {"Start": ["Target", "Other"], "Target": [], "Other": []},
        )
        episode = collect_episode(graph, Task("Start", "Target"), FixedPolicy(),
                                  reward_config(), "episode-test", policy_version=4)
        self.assertTrue(episode["success"])
        self.assertEqual(episode["steps"], 1)
        self.assertAlmostEqual(episode["return"], 0.98)
        self.assertEqual(len(episode["transitions"]), 1)
        self.assertEqual(episode["transitions"][0]["reward"], 0.98)
        self.assertEqual(len(episode["transitions"][0]["nodes"]), 1)
        self.assertEqual(episode["transitions"][0]["nodes"][0]["old_log_prob"], -0.5)

    def test_zero_click_dead_end_is_recorded_as_failure(self):
        graph = WikiGraph({"Start", "Target"}, {"Start": [], "Target": []})
        episode = collect_episode(graph, Task("Start", "Target"), FixedPolicy(), reward_config())
        self.assertEqual(episode["steps"], 0)
        self.assertEqual(episode["termination_reason"], "dead_end")
        self.assertEqual(episode["return"], -1.0)
        self.assertEqual(episode["transitions"], [])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "training extra is not installed")
    def test_evaluation_streams_completed_tasks_before_later_failure(self):
        import torch

        from wiki_speedrun.training.config import DataConfig, ModelConfig, TrainingConfig
        from wiki_speedrun.training.rl import evaluate_trained

        class FailingScorer:
            training = True

            def eval(self):
                self.training = False
                return self

            def train(self, mode=True):
                self.training = mode
                return self

            def state_token_count(self, state):
                return 1

            def fits(self, request):
                return True

            def encode(self, request):
                return {
                    "choice_ids": list(request["questions"]["next_link"]["criteria"]),
                    "request": request,
                }

            def logits(self, encoded):
                if encoded["request"]["state"]["current_title"] == "Start Two":
                    raise RuntimeError("injected second-task failure")
                return torch.tensor([1.0, 0.0])

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dev_path = root / "dev.json"
            pairs = [
                {"start": "Start_One", "target": "Target_One"},
                {"start": "Start_Two", "target": "Target_Two"},
            ]
            dev_path.write_text(json.dumps(pairs), encoding="utf-8")
            config = TrainingConfig(
                data_dir=str(root), run_dir=str(root / "runs"),
                model=ModelConfig(device="cpu", autocast_dtype="fp32"),
                data=DataConfig(dev_tasks=str(dev_path)),
            )
            graph = WikiGraph(
                {"Start_One", "Target_One", "Decoy_One", "Start_Two", "Target_Two", "Decoy_Two"},
                {"Start_One": ["Target_One", "Decoy_One"], "Start_Two": ["Target_Two", "Decoy_Two"],
                 "Target_One": [], "Decoy_One": [], "Target_Two": [], "Decoy_Two": []},
            )
            scorer = FailingScorer()
            previous_cwd = Path.cwd()
            os.chdir(root)
            try:
                with self.assertRaisesRegex(RuntimeError, "second-task failure"):
                    evaluate_trained(config, "fake-checkpoint", graph=graph, model=scorer)
                self.assertTrue(scorer.training)
                run_dir = next((root / "runs").iterdir())
                transitions = [json.loads(line) for line in
                               (run_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines()]
                episodes = [json.loads(line) for line in
                            (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
                error = json.loads((run_dir / "error.json").read_text(encoding="utf-8"))
                self.assertEqual(len(transitions), 1)
                self.assertEqual(len(episodes), 1)
                self.assertIn("request", transitions[0]["nodes"][0])
                self.assertEqual(error["last_completed_task_index"], 0)
                self.assertEqual(error["task"], {"start": "Start_Two", "target": "Target_Two"})
            finally:
                os.chdir(previous_cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "training extra is not installed")
    def test_tiny_reinforce_update_logs_clicks_and_exports_checkpoint(self):
        import torch

        from wiki_speedrun.training.config import (
            ChoiceConfig, DataConfig, ModelConfig, RewardConfig, RlConfig,
            SaveConfig, SftConfig, TrainingConfig,
        )
        from wiki_speedrun.training.rl import train_rl

        class TinyScorer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(2))
                self.metadata = {"fake": True}
                self.grad_requests = []

            def encode(self, request):
                return {"choice_ids": list(request["questions"]["next_link"]["criteria"])}

            def logits(self, encoded):
                if torch.is_grad_enabled():
                    self.grad_requests.append(tuple(encoded["choice_ids"]))
                return self.bias[:len(encoded["choice_ids"])].float()

            def state_token_count(self, state):
                return 1

            def fits(self, request):
                return True

            def save_checkpoint(self, directory):
                torch.save(self.state_dict(), Path(directory) / "tiny-model.pt")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train_tasks = root / "train.json"
            dev_tasks = root / "dev.json"
            test_tasks = root / "test.json"
            manifest = root / "manifest.json"
            pair = [{"start": "Start", "target": "Target"}]
            train_tasks.write_text(json.dumps(pair), encoding="utf-8")
            dev_tasks.write_text(json.dumps(pair), encoding="utf-8")
            test_tasks.write_text(json.dumps(pair), encoding="utf-8")
            manifest.write_text("{}\n", encoding="utf-8")
            config = TrainingConfig(
                data_dir=str(root), run_dir=str(root / "runs"),
                model=ModelConfig(device="cpu", autocast_dtype="fp32"),
                choices=ChoiceConfig(node_max_choices=2),
                data=DataConfig(split_manifest=str(manifest),
                                rl_train_tasks=str(train_tasks), dev_tasks=str(dev_tasks),
                                test_tasks=str(test_tasks), sft_train=str(root / "sft.jsonl"),
                                sft_dev=str(root / "sft-dev.jsonl")),
                sft=SftConfig(),
                rl=RlConfig(updates=1, task_pairs_per_update=1, episodes_per_pair=2),
                reward=RewardConfig(),
                save=SaveConfig(checkpoint_every_updates=1, evaluate_every_updates=1),
            )
            config.save.evaluation_max_tasks = 1
            graph = WikiGraph({"Start", "Target", "Other1", "Other2", "Other3"},
                              {"Start": ["Target", "Other1", "Other2", "Other3"],
                               "Target": [], "Other1": [], "Other2": [], "Other3": []})
            previous_cwd = Path.cwd()
            os.chdir(root)
            try:
                scorer = TinyScorer()
                checkpoint = train_rl(config, max_updates=1, graph=graph,
                                      model=scorer)
                self.assertTrue((checkpoint / "training_state.pt").is_file())
                self.assertTrue((checkpoint / "manifest.json").is_file())
                run_dir = next((root / "runs").iterdir())
                transitions = [json.loads(line) for line in
                               (run_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines()]
                episodes = [json.loads(line) for line in
                            (run_dir / "episodes.jsonl").read_text(encoding="utf-8").splitlines()]
                self.assertEqual(len(episodes), 2)
                self.assertEqual(len(transitions), 2)
                self.assertTrue(all(len(item["nodes"]) == 3 for item in transitions))
                expected_nodes = Counter(
                    tuple(node["candidate_ids"])
                    for transition in transitions for node in transition["nodes"]
                )
                self.assertEqual(Counter(scorer.grad_requests), expected_nodes)
            finally:
                os.chdir(previous_cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "training extra is not installed")
    def test_two_update_rl_resume_matches_task_sampling_nodes_and_weights(self):
        import torch

        from wiki_speedrun.training.config import (
            ChoiceConfig, DataConfig, ModelConfig, RewardConfig, RlConfig,
            SaveConfig, SftConfig, TrainingConfig,
        )
        from wiki_speedrun.training.rl import train_rl

        class TinyScorer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(2))
                self.metadata = {"fake": True}

            def encode(self, request):
                return {"choice_ids": list(request["questions"]["next_link"]["criteria"])}

            def logits(self, encoded):
                return self.bias[:len(encoded["choice_ids"])].float()

            def state_token_count(self, state):
                return 1

            def fits(self, request):
                return True

            def save_checkpoint(self, directory):
                torch.save(self.state_dict(), Path(directory) / "tiny-model.pt")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {
                "train": root / "train.json",
                "dev": root / "dev.json",
                "test": root / "test.json",
                "manifest": root / "manifest.json",
                "sft_train": root / "sft-train.jsonl",
                "sft_dev": root / "sft-dev.jsonl",
            }
            pairs = [
                {"start": "StartA", "target": "TargetA"},
                {"start": "StartB", "target": "TargetB"},
            ]
            for name in ("train", "dev", "test"):
                paths[name].write_text(json.dumps(pairs), encoding="utf-8")
            paths["manifest"].write_text("{}\n", encoding="utf-8")
            paths["sft_train"].write_text("", encoding="utf-8")
            paths["sft_dev"].write_text("", encoding="utf-8")
            config = TrainingConfig(
                data_dir=str(root), run_dir=str(root / "runs"),
                model=ModelConfig(device="cpu", autocast_dtype="fp32"),
                choices=ChoiceConfig(node_max_choices=2),
                data=DataConfig(split_manifest=str(paths["manifest"]),
                                rl_train_tasks=str(paths["train"]), dev_tasks=str(paths["dev"]),
                                test_tasks=str(paths["test"]), sft_train=str(paths["sft_train"]),
                                sft_dev=str(paths["sft_dev"])),
                sft=SftConfig(),
                rl=RlConfig(updates=2, task_pairs_per_update=2, episodes_per_pair=2),
                reward=RewardConfig(),
                save=SaveConfig(checkpoint_every_updates=1, evaluate_every_updates=1),
            )
            graph = WikiGraph(
                {"StartA", "StartB", "TargetA", "TargetB", "A1", "A2", "A3", "B1", "B2", "B3"},
                {"StartA": ["TargetA", "A1", "A2", "A3"],
                 "StartB": ["TargetB", "B1", "B2", "B3"],
                 "TargetA": [], "TargetB": [], "A1": [], "A2": [], "A3": [],
                 "B1": [], "B2": [], "B3": []},
            )
            torch.manual_seed(19)
            initial = TinyScorer().state_dict()
            full_root = root / "full"
            split_root = root / "split"
            full_root.mkdir()
            split_root.mkdir()
            previous_cwd = Path.cwd()
            try:
                os.chdir(full_root)
                full_model = TinyScorer()
                full_model.load_state_dict(initial)
                full_checkpoint = train_rl(config, max_updates=2,
                                           graph=graph, model=full_model).resolve()
                full_run_dir = root / "runs" / full_checkpoint.parent.name
                full_transitions = [json.loads(line) for line in
                                    (full_run_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines()]

                os.chdir(split_root)
                first_model = TinyScorer()
                first_model.load_state_dict(initial)
                first_checkpoint = train_rl(config, max_updates=1,
                                            graph=graph, model=first_model).resolve()
                resumed_model = TinyScorer()
                resumed_model.load_state_dict(torch.load(first_checkpoint / "tiny-model.pt",
                                                          weights_only=True))
                resumed_checkpoint = train_rl(
                    config, resume=first_checkpoint, max_updates=2,
                    graph=graph, model=resumed_model,
                ).resolve()
                resumed_run_dir = root / "runs" / resumed_checkpoint.parent.name
                resumed_transitions = [json.loads(line) for line in
                                       (resumed_run_dir / "transitions.jsonl").read_text(encoding="utf-8").splitlines()]

                full_second_update = [item for item in full_transitions if item["policy_version"] == 1]
                resumed_second_update = [item for item in resumed_transitions if item["policy_version"] == 1]
                for transition in full_second_update + resumed_second_update:
                    transition.pop("latency_ms", None)
                self.assertEqual(full_second_update, resumed_second_update)
                full_weights = torch.load(full_checkpoint / "tiny-model.pt", weights_only=True)
                resumed_weights = torch.load(resumed_checkpoint / "tiny-model.pt", weights_only=True)
                for name, value in full_weights.items():
                    self.assertTrue(torch.equal(value, resumed_weights[name]))
            finally:
                os.chdir(previous_cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "training extra is not installed")
    def test_tiny_sft_update_uses_cross_entropy_and_saves_best(self):
        import torch

        from wiki_speedrun.training.config import (
            ChoiceConfig, DataConfig, ModelConfig, RewardConfig, RlConfig,
            SaveConfig, SftConfig, TrainingConfig,
        )
        from wiki_speedrun.training.rl import evaluate_trained
        from wiki_speedrun.training.sft import train_sft

        class TinyScorer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = torch.nn.Parameter(torch.zeros(2))
                self.metadata = {"fake": True}

            def encode(self, request):
                return {"choice_ids": list(request["questions"]["next_link"]["criteria"])}

            def logits(self, encoded):
                return self.bias[:len(encoded["choice_ids"])].float()

            def state_token_count(self, state):
                return 1

            def fits(self, request):
                return True

            def save_checkpoint(self, directory):
                torch.save(self.state_dict(), Path(directory) / "tiny-model.pt")

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train_path = root / "sft-train.jsonl"
            dev_sft_path = root / "sft-dev.jsonl"
            train_tasks = root / "train.json"
            dev_tasks = root / "dev.json"
            test_tasks = root / "test.json"
            manifest = root / "manifest.json"
            request = {
                "model": "kev-latest", "state": {"current_title": "Start"},
                "questions": {"next_link": {
                    "type": "choice", "instructions": "Pick the route.",
                    "criteria": {"link_0": "Target", "link_1": "Other"},
                }},
            }
            record = {"request": request, "candidate_ids": ["link_0", "link_1"],
                      "articles": ["Target", "Other"], "label": "link_0"}
            line = json.dumps(record) + "\n"
            train_path.write_text(line, encoding="utf-8")
            dev_sft_path.write_text(line, encoding="utf-8")
            pair = [{"start": "Start", "target": "Target"}]
            dev_pairs = pair + [{"start": "Start", "target": "Other"}]
            train_tasks.write_text(json.dumps(pair), encoding="utf-8")
            dev_tasks.write_text(json.dumps(dev_pairs), encoding="utf-8")
            test_tasks.write_text(json.dumps(pair), encoding="utf-8")
            manifest.write_text("{}\n", encoding="utf-8")
            config = TrainingConfig(
                data_dir=str(root), run_dir=str(root / "runs"),
                model=ModelConfig(device="cpu", autocast_dtype="fp32"),
                choices=ChoiceConfig(node_max_choices=2),
                data=DataConfig(split_manifest=str(manifest),
                                sft_train=str(train_path), sft_dev=str(dev_sft_path),
                                rl_train_tasks=str(train_tasks), dev_tasks=str(dev_tasks),
                                test_tasks=str(test_tasks)),
                sft=SftConfig(epochs=3, max_updates=1, gradient_accumulation=1),
                rl=RlConfig(), reward=RewardConfig(),
                save=SaveConfig(checkpoint_every_updates=1, evaluate_every_updates=1),
            )
            config.save.evaluation_max_tasks = 1
            graph = WikiGraph({"Start", "Target", "Other"},
                              {"Start": ["Target", "Other"], "Target": [], "Other": []})
            previous_cwd = Path.cwd()
            try:
                torch.manual_seed(77)
                initial_model = TinyScorer()
                initial_weights = {key: value.detach().clone()
                                   for key, value in initial_model.state_dict().items()}

                uninterrupted_root = root / "uninterrupted"
                interrupted_root = root / "interrupted"
                uninterrupted_root.mkdir()
                interrupted_root.mkdir()

                os.chdir(uninterrupted_root)
                uninterrupted_model = TinyScorer()
                uninterrupted_model.load_state_dict(initial_weights)
                uninterrupted_checkpoint = train_sft(
                    config, max_updates=2, max_examples=1,
                    graph=graph, model=uninterrupted_model,
                ).resolve()

                os.chdir(interrupted_root)
                first_model = TinyScorer()
                first_model.load_state_dict(initial_weights)
                first_checkpoint = train_sft(config, max_examples=1,
                                             graph=graph, model=first_model).resolve()
                first_state = torch.load(first_checkpoint / "training_state.pt",
                                         weights_only=False)
                self.assertEqual(first_state["completed_update"], 1)
                self.assertLess(first_state["cursor"]["epoch"], config.sft.epochs)
                resumed_model = TinyScorer()
                resumed_model.load_state_dict(torch.load(first_checkpoint / "tiny-model.pt",
                                                          weights_only=True))
                resumed_checkpoint = train_sft(
                    config, resume=first_checkpoint, max_updates=2,
                    max_examples=1, graph=graph, model=resumed_model,
                ).resolve()
                self.assertTrue((resumed_checkpoint / "training_state.pt").is_file())
                self.assertTrue((resumed_checkpoint.parent / "best.json").is_file())

                expected = torch.load(uninterrupted_checkpoint / "tiny-model.pt",
                                      weights_only=True)
                actual = torch.load(resumed_checkpoint / "tiny-model.pt",
                                    weights_only=True)
                for key, value in expected.items():
                    self.assertTrue(torch.equal(value, actual[key]))
                metrics_path = root / "runs" / resumed_checkpoint.parent.name / "metrics.jsonl"
                committed_size = metrics_path.stat().st_size
                no_op_checkpoint = train_sft(
                    config, resume=resumed_checkpoint, max_updates=2,
                    max_examples=1, graph=graph, model=TinyScorer(),
                )
                self.assertEqual(no_op_checkpoint.resolve(), resumed_checkpoint)
                self.assertEqual(metrics_path.stat().st_size, committed_size)
                evaluation = evaluate_trained(
                    config, resumed_checkpoint, split="dev",
                    graph=graph, model=resumed_model,
                )
                eval_dir = Path(evaluation["run_dir"])
                self.assertEqual(evaluation["episode_count"], 2)
                self.assertTrue((eval_dir / "provenance.json").is_file())
                self.assertTrue((eval_dir / "metrics.json").is_file())
                self.assertEqual(len((eval_dir / "episodes.jsonl").read_text(
                    encoding="utf-8").splitlines()), 2)
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
