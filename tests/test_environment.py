import unittest

from wiki_speedrun.data.load import WikiGraph
from wiki_speedrun.data.tasks import Task
from wiki_speedrun.environment import NavigationEnvironment
from wiki_speedrun.observations import build_request
from wiki_speedrun.policy import KevPolicy
from wiki_speedrun.runner import run_episode


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.graph = WikiGraph(
            articles={"A", "B", "C", "D"},
            links={"A": ["B", "C"], "B": ["D"], "C": [], "D": []},
            texts={"A": "start excerpt", "B": "middle"},
        )

    def test_moves_and_success(self):
        env = NavigationEnvironment(self.graph, "A", "D")
        self.assertEqual(env.legal_moves(), ["B", "C"])
        with self.assertRaises(ValueError):
            env.step("D")
        env.step("B")
        env.step("D")
        self.assertTrue(env.done)
        self.assertTrue(env.success)
        self.assertEqual(env.termination_reason, "success")

    def test_dead_end_and_step_limit(self):
        dead = NavigationEnvironment(self.graph, "C", "D")
        self.assertEqual(dead.termination_reason, "dead_end")
        limited = NavigationEnvironment(self.graph, "A", "D", max_steps=1)
        limited.step("B")
        self.assertEqual(limited.termination_reason, "step_limit")
        with self.assertRaises(RuntimeError):
            limited.step("D")

    def test_request_overflow_is_rejected(self):
        env = NavigationEnvironment(self.graph, "A", "D")
        with self.assertRaises(ValueError):
            build_request(env, max_options=1)

    def test_observation_does_not_expose_graph_distances_or_target_text(self):
        self.graph.texts["D"] = "secret target text"
        request = build_request(NavigationEnvironment(self.graph, "A", "D"))
        self.assertEqual(request["state"]["current_excerpt"], "start excerpt")
        self.assertNotIn("secret target text", str(request))
        self.assertNotIn("distance", str(request).lower())

    def test_runner_records_trace(self):
        class FirstLinkPolicy:
            def choose(self, env):
                return {"article": env.legal_moves()[0], "probabilities": {}, "confidence": None, "latency_ms": 0}

        result = run_episode(self.graph, Task("A", "D"), FirstLinkPolicy())
        self.assertTrue(result["success"])
        self.assertEqual(result["path"], ["A", "B", "D"])
        self.assertEqual(result["steps"], 2)
        self.assertEqual(len(result["decisions"]), 2)
        self.assertEqual(result["decisions"][0]["source"], "A")
        self.assertEqual(result["decisions"][0]["options"], {"link_0": "B", "link_1": "C"})

    def test_runner_reports_option_limit_with_partial_trace(self):
        class Client:
            def score(self, request):
                return {"choice": "link_0", "probabilities": {"link_0": 1.0}, "confidence": 1.0}

        graph = WikiGraph(
            articles={"X", "A", "B", "C", "D"},
            links={"X": ["A"], "A": ["B", "C"], "B": ["D"], "C": [], "D": []},
            texts={},
        )
        policy = KevPolicy(Client(), max_options=1)
        result = run_episode(graph, Task("X", "D"), policy)
        self.assertEqual(result["termination_reason"], "option_limit")
        self.assertEqual(result["path"], ["X", "A"])
        self.assertEqual(len(result["decisions"]), 1)
        self.assertIn("exceed max_options", result["error_message"])


if __name__ == "__main__":
    unittest.main()
