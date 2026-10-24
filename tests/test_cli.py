"""Exercise the installed CLI from a prepared dataset through saved results."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from wiki_speedrun.cli import main


class CliTests(unittest.TestCase):
    def test_benchmark_with_training_config_routes_to_benchmark(self):
        commands = [
            ["benchmark"],
            ["benchmark", "--training-config", "configs/snap-full.toml",
             "--models", "minimax", "deepseek", "jev", "--splits", "dev",
             "--remote-workers", "3", "--no-include-local", "--max-total-cost", "9",
             "--output", "runs/snap-full/remote-dev"],
        ]
        for command in commands:
            with self.subTest(command=command), \
                 patch("wiki_speedrun.benchmark.benchmark_command", return_value=0) as benchmark, \
                 patch("wiki_speedrun.cli.training_command") as training:
                self.assertEqual(main(command), 0)
                benchmark.assert_called_once()
                training.assert_not_called()
                args = benchmark.call_args.args[0]
                self.assertEqual(args.command, "benchmark")
                if "--training-config" in command:
                    self.assertEqual(args.training_config, Path("configs/snap-full.toml"))
                    self.assertEqual(args.models, ["minimax", "deepseek", "jev"])
                    self.assertEqual(args.splits, ["dev"])
                    self.assertEqual(args.remote_workers, 3)
                    self.assertFalse(args.include_local)
                    self.assertEqual(args.max_total_cost, 9)
                    self.assertEqual(args.output, Path("runs/snap-full/remote-dev"))

    def test_training_commands_still_route_to_training(self):
        commands = [
            ["prepare-training"], ["train-sft"], ["train-rl"],
            ["evaluate-trained", "--checkpoint", "checkpoints/example"],
        ]
        for command in commands:
            with self.subTest(command=command), \
                 patch("wiki_speedrun.cli.training_command", return_value=0) as training:
                self.assertEqual(main(command), 0)
                self.assertEqual(training.call_args.args[0].command, command[0])

    def test_play_saves_a_valid_complete_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data" / "processed"
            data.mkdir(parents=True)
            (data / "graph.json").write_text(json.dumps({
                "articles": ["A", "B", "C"],
                "links": {"A": ["B"], "B": ["C"], "C": []},
                "texts": {},
            }), encoding="utf-8")
            config = root / "test.toml"
            config.write_text(
                f'data_dir = "{(root / "data").as_posix()}"\n'
                f'run_dir = "{(root / "runs").as_posix()}"\n',
                encoding="utf-8",
            )
            with redirect_stdout(io.StringIO()):
                result = main(["--config", str(config), "play", "A", "C", "--policy", "random"])
            self.assertEqual(result, 0)
            folder = next((root / "runs").iterdir())
            episode = json.loads((folder / "episodes.jsonl").read_text(encoding="utf-8"))
            self.assertEqual(episode["path"], ["A", "B", "C"])
            metrics = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(metrics["success_rate"], 1.0)
            self.assertEqual(metrics["mean_shortest_path_overhead"], 0)
