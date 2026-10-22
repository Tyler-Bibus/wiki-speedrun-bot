"""Catch settings that would silently change training or make rewards unsafe."""

from pathlib import Path
import tempfile
import unittest

from wiki_speedrun.cli import main, parser
from wiki_speedrun.training.config import (
    ChoiceConfig, RewardConfig, RlConfig, TrainingConfig, load_training_config,
)


class TrainingConfigTests(unittest.TestCase):
    def test_committed_config_matches_defaults(self):
        self.assertEqual(load_training_config(Path("configs/training.toml")), TrainingConfig())

    def test_unknown_keys_and_nonfinite_values_fail(self):
        examples = [
            'sead = 42\n',
            '[rl]\nlearning_rates = 0.001\n',
            '[rl]\nlearning_rate = nan\n',
            '[reward]\nstep_cost = inf\n',
            '[choices]\nnode_max_choices = 2.5\n',
            '[data]\nmax_paths = 0\n',
            '[sft]\nmax_updates = 0\n',
            '[save]\nkeep_last_checkpoints = 0\n',
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            for content in examples:
                with self.subTest(content=content):
                    path.write_text(content)
                    with self.assertRaises(ValueError):
                        load_training_config(path)

    def test_success_must_beat_failure_at_maximum_path_length(self):
        with self.assertRaisesRegex(ValueError, "success must beat"):
            TrainingConfig(reward=RewardConfig(step_cost=0.2))

    def test_tournament_must_reduce_and_baseline_needs_other_episodes(self):
        with self.assertRaisesRegex(ValueError, "between 2 and 255"):
            TrainingConfig(choices=ChoiceConfig(node_max_choices=1))
        with self.assertRaisesRegex(ValueError, "at least 2"):
            TrainingConfig(rl=RlConfig(episodes_per_pair=1))

    def test_baseline_and_training_config_flags_remain_separate(self):
        baseline = parser().parse_args(["--config", "baseline.toml", "play", "A", "B"])
        training = parser().parse_args(["train-sft", "--config", "training.toml"])
        self.assertEqual(baseline.config, Path("baseline.toml"))
        self.assertEqual(training.training_config, Path("training.toml"))

    def test_invalid_smoke_limit_fails_before_loading_gpu_dependencies(self):
        self.assertEqual(main(["train-sft", "--max-updates", "0"]), 1)
