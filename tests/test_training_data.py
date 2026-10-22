import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from wiki_speedrun.data.training import (
    _preflight_graph, _sha256_file, prepare_training, validate_prepared_data,
)
from wiki_speedrun.data.load import WikiGraph
from wiki_speedrun.training.config import (
    ChoiceConfig, DataConfig, ModelConfig, RlConfig, TrainingConfig,
)


class FakeTokenizer:
    vocab = {"a": 0, "b": 1}
    special_tokens_map = {"bos_token": "<s>"}


class FakeModel:
    tokenizer = FakeTokenizer()
    resolved_commit = "test-commit"
    metadata = {"base_revision": "test-commit"}

    def state_token_count(self, state):
        return len(state.get("current_excerpt", "").split()) + len(state.get("history", []))

    def fits(self, request):
        return len(request["questions"]["next_link"]["criteria"]) <= 2

    def option_token_count(self, choice_id, title):
        return len(choice_id) + len(title) + 2

    def instruction_token_count(self, instructions):
        return 3


class TrainingDataTests(unittest.TestCase):
    def make_config(self, root):
        return TrainingConfig(
            seed=7,
            data_dir=str(root / "data"),
            model=ModelConfig(device="cpu"),
            choices=ChoiceConfig(node_max_choices=4, max_state_tokens=16,
                                 max_row_tokens=64, excerpt_chars=50),
            data=DataConfig(
                split_manifest=str(root / "out/manifest.json"),
                sft_train=str(root / "out/sft_train.jsonl"),
                sft_dev=str(root / "out/sft_dev.jsonl"),
                rl_train_tasks=str(root / "out/train_tasks.json"),
                dev_tasks=str(root / "out/dev_tasks.json"),
                test_tasks=str(root / "out/test_tasks.json"),
            ),
        )

    def test_pair_splits_outputs_are_deterministic_and_disjoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            (data / "processed").mkdir(parents=True)
            (data / "raw" / "snap").mkdir(parents=True)
            graph = {
                "articles": ["A", "B", "C", "D"],
                "links": {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []},
                "texts": {"A": "excerpt for A"},
            }
            (data / "processed/graph.json").write_text(json.dumps(graph), encoding="utf-8")
            paths = data / "raw/snap/paths_finished.tsv"
            paths.write_text("A;B;D\nA;C;D\nB;D\nC;D\nA;Unknown\n", encoding="utf-8")
            config = self.make_config(root)
            first = prepare_training(config, FakeModel())
            second = prepare_training(config, FakeModel())
            self.assertEqual(first, second)
            self.assertEqual(validate_prepared_data(config, FakeModel()), first)
            self.assertEqual(first["rejection_counts"]["unknown_article"], 1)
            assignments = {(row["start"], row["target"]): row["split"]
                           for row in first["pair_assignments"]}
            train = json.loads(Path(config.data.rl_train_tasks).read_text(encoding="utf-8"))
            dev = json.loads(Path(config.data.dev_tasks).read_text(encoding="utf-8"))
            test = json.loads(Path(config.data.test_tasks).read_text(encoding="utf-8"))
            pair_sets = [{(row["start"], row["target"]) for row in items}
                         for items in (train, dev, test)]
            self.assertFalse(pair_sets[0] & pair_sets[1])
            self.assertFalse(pair_sets[0] & pair_sets[2])
            self.assertFalse(pair_sets[1] & pair_sets[2])
            self.assertTrue(all(assignments[pair] == split for split, pairs in zip(
                ("train", "dev", "test"), pair_sets) for pair in pairs))
            lines = Path(config.data.sft_train).read_text(encoding="utf-8").splitlines()
            examples = [json.loads(line) for line in lines]
            for item in examples:
                self.assertIn(item["label"], item["candidate_ids"])
                self.assertEqual(item["request"]["questions"]["next_link"]["criteria"].keys(),
                                 set(item["candidate_ids"]))
            assignments_by_pair = {(item["start"], item["target"]): item
                                   for item in first["pair_assignments"]}
            self.assertEqual(assignments_by_pair[("A", "D")]["shortest_distance"], 2)
            self.assertEqual(assignments_by_pair[("A", "D")]["source"], ["human"])

    def test_validator_checks_config_model_hashes_and_split_membership(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            (data / "processed").mkdir(parents=True)
            (data / "raw/snap").mkdir(parents=True)
            (data / "processed/graph.json").write_text(json.dumps({
                "articles": ["A", "B", "C", "D"],
                "links": {"A": ["B", "C"], "B": ["D"], "C": ["D"], "D": []},
            }), encoding="utf-8")
            (data / "raw/snap/paths_finished.tsv").write_text(
                "A;B;D\nA;C;D\nB;D\nC;D\n", encoding="utf-8")
            config = self.make_config(root)
            manifest = prepare_training(config, FakeModel())

            wrong_model = FakeModel()
            wrong_model.metadata = {"base_revision": "different-commit"}
            with self.assertRaisesRegex(ValueError, "base model revision"):
                validate_prepared_data(config, wrong_model)

            original_excerpt = config.choices.excerpt_chars
            config.choices.excerpt_chars += 1
            with self.assertRaisesRegex(ValueError, "choice_config"):
                validate_prepared_data(config)
            config.choices.excerpt_chars = original_excerpt

            train_tasks = json.loads(Path(config.data.rl_train_tasks).read_text(encoding="utf-8"))
            dev_path = Path(config.data.dev_tasks)
            dev_tasks = json.loads(dev_path.read_text(encoding="utf-8"))
            dev_tasks.append(train_tasks[0])
            dev_path.write_text(json.dumps(dev_tasks), encoding="utf-8")
            manifest["output_sha256"][str(dev_path)] = _sha256_file(dev_path)
            Path(config.data.split_manifest).write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not assigned to dev"):
                validate_prepared_data(config)

            # A hash mismatch is rejected before task content is trusted.
            dev_path.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "output hash mismatch"):
                validate_prepared_data(config)

    def test_bounded_selection_is_fingerprinted_and_incompatible_manifest_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            (data / "processed").mkdir(parents=True)
            (data / "raw/snap").mkdir(parents=True)
            (data / "processed/graph.json").write_text(json.dumps({
                "articles": ["A", "B", "C"],
                "links": {"A": ["B", "C"], "B": ["C"], "C": []},
            }), encoding="utf-8")
            (data / "raw/snap/paths_finished.tsv").write_text(
                "A;B\nA;C\nB;C\n", encoding="utf-8")
            config = self.make_config(root)
            config.data.max_paths = 2
            manifest = prepare_training(config, FakeModel())
            self.assertEqual(len(manifest["selected_source_row_ids"]), 2)
            self.assertEqual(manifest["max_paths"], 2)
            config.data.max_paths = 1
            self.assertEqual(prepare_training(config, FakeModel(), max_paths=2), manifest)
            config.seed += 1
            with self.assertRaises(FileExistsError):
                prepare_training(config, FakeModel(), max_paths=2)

    def test_preflight_uses_worst_two_choices_and_fails_clear(self):
        graph = WikiGraph({"Start", "Target"}, {"Start": ["Target", "Target"], "Target": []})
        config = SimpleNamespace(choices=ChoiceConfig(max_state_tokens=8, max_row_tokens=64))
        result = _preflight_graph(FakeModel(), graph, config)
        self.assertTrue(result["available"])
        self.assertEqual(result["checked_pages"], 1)
        config.choices.max_row_tokens = 20
        with self.assertRaisesRegex(ValueError, "token preflight failed"):
            _preflight_graph(FakeModel(), graph, config)


if __name__ == "__main__":
    unittest.main()
