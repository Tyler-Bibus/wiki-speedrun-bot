"""Exercise the installed CLI from a prepared dataset through saved results."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from wiki_speedrun.cli import main


class CliTests(unittest.TestCase):
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
