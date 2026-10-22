import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from wiki_speedrun.data.download import _safe_extract
from wiki_speedrun.data.load import WikiGraph, display_title, load_graph, load_paths, load_raw_graph
from wiki_speedrun.data.tasks import Task, load_tasks, make_tasks, save_tasks, shortest_path


class DataTests(unittest.TestCase):
    def test_parse_raw_and_processed_graph(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "articles.tsv").write_text("# comment\nNew_York\tCity\nCaf%C3%A9\tFood\n", encoding="utf-8")
            (root / "links.tsv").write_text("# comment\nNew_York\tCaf%C3%A9\n", encoding="utf-8")
            graph = load_raw_graph(root)
            self.assertEqual(graph.articles, {"New_York", "Caf" + chr(233)})
            self.assertEqual(graph.links["New_York"], ["Caf" + chr(233)])
            self.assertEqual(display_title("New_York"), "New York")
            output = root / "processed" / "graph.json"
            output.parent.mkdir()
            output.write_text(json.dumps({"articles": list(graph.articles), "links": graph.links}), encoding="utf-8")
            self.assertEqual(load_graph(root).articles, graph.articles)

    def test_paths_expand_back_marker(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "paths.tsv"
            path.write_text("# header\nhash\ttimestamp\tduration\tA;B;C;<;D\trating\n", encoding="utf-8")
            self.assertEqual(load_paths(path), [["A", "B", "D"]])

    def test_tasks_use_full_graph_distance_and_bounded_route(self):
        graph = WikiGraph(
            {"A", "B", "C", "H", "T", "X", "Y"},
            {"A": ["H", "B"], "H": ["T", "X", "Y"], "B": ["C"],
             "C": ["T"], "T": [], "X": [], "Y": []},
        )
        self.assertEqual(len(shortest_path(graph, "A", "T")) - 1, 2)
        tasks = make_tasks(graph, count=20, min_distance=2, max_distance=3,
                           seed=9, max_options=2)
        self.assertIn("A", {task.start for task in tasks})
        self.assertEqual(len(tasks), len({task.start for task in tasks}))
        self.assertTrue(all(2 <= len(shortest_path(graph, task.start, task.target)) - 1 <= 3
                            for task in tasks))
        self.assertEqual(tasks, make_tasks(graph, count=20, min_distance=2,
                                           max_distance=3, seed=9, max_options=2))

    def test_download_failure_does_not_cache_partial_file(self):
        from wiki_speedrun.data.download import _download
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "archive.tar.gz"
            with patch("wiki_speedrun.data.download.urllib.request.urlopen", side_effect=OSError("network")):
                with self.assertRaises(OSError):
                    _download("https://example.invalid/archive.tar.gz", target)
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(temp).glob("*.part")), [])

    def test_safe_extract_ignores_traversal_and_links(self):
        import io
        import tarfile
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / "test.tar.gz"
            with tarfile.open(archive, "w:gz") as tar:
                info = tarfile.TarInfo("../escape.txt")
                payload = b"bad"
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
                info = tarfile.TarInfo("safe/file.txt")
                info.size = 2
                tar.addfile(info, io.BytesIO(b"ok"))
            _safe_extract(archive, root / "out")
            self.assertFalse((root / "escape.txt").exists())
            self.assertEqual((root / "out" / "safe" / "file.txt").read_bytes(), b"ok")

    def test_paths_and_task_roundtrip(self):
        graph = WikiGraph({"A", "B", "C", "D"}, {"A": ["B", "C"], "B": ["C"], "C": ["D"], "D": []})
        self.assertEqual(shortest_path(graph, "A", "D"), ["A", "C", "D"])
        self.assertIsNone(shortest_path(graph, "D", "A"))
        tasks = make_tasks(graph, count=10, min_distance=2, max_distance=2, seed=3)
        self.assertTrue(tasks)
        self.assertTrue(all(2 <= len(shortest_path(graph, task.start, task.target)) - 1 <= 2 for task in tasks))
        self.assertEqual(make_tasks(WikiGraph({"solo"}, {"solo": []})), [])
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tasks.json"
            save_tasks([Task("A", "D")], path)
            self.assertEqual(load_tasks(path), [Task("A", "D")])


if __name__ == "__main__":
    unittest.main()
