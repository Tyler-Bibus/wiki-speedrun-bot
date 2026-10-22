import unittest
from types import SimpleNamespace
from unittest.mock import patch

from wiki_speedrun.choices import TournamentPolicy, build_supervised_nodes
from wiki_speedrun.data.load import WikiGraph
from wiki_speedrun.environment import NavigationEnvironment


class FakeModel:
    def __init__(self, max_candidates=2):
        self.max_candidates = max_candidates

    def state_token_count(self, state):
        return len(state.get("history", [])) + len(state.get("current_excerpt", ""))

    def fits(self, request):
        return len(request["questions"]["next_link"]["criteria"]) <= self.max_candidates


class ChoiceTests(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(node_max_choices=4, max_state_tokens=32,
                                      excerpt_chars=10, max_row_tokens=64)
        self.model = FakeModel()

    def test_tournament_covers_original_ids_and_keeps_duplicate_edges(self):
        graph = WikiGraph({"Start", "A", "B", "C", "Target"}, {
            "Start": ["A", "A", "B", "C", "Target", "Target"],
            "A": [], "B": [], "C": [], "Target": [],
        }, {"Start": "excerpt"})
        env = NavigationEnvironment(graph, "Start", "Target")
        policy = TournamentPolicy(self.model, self.config)
        with patch("wiki_speedrun.choices._predict", side_effect=lambda model, request, ids, sample:
                   (ids[0], None, 0.0)):
            result = policy.choose(env)
        first_round = [node for node in result["nodes"] if node["round"] == 0]
        covered = [candidate for node in first_round for candidate in node["candidate_ids"]]
        self.assertEqual(covered, ["link_0", "link_1", "link_2", "link_3", "link_4", "link_5"])
        self.assertEqual(first_round[0]["articles"], ["A", "A"])
        self.assertEqual(result["selected_link_id"], "link_0")
        self.assertEqual(result["article"], "A")

    def test_single_move_is_passed_through_without_a_model_call(self):
        graph = WikiGraph({"Start", "Target"}, {"Start": ["Target"], "Target": []})
        policy = TournamentPolicy(self.model, self.config)
        with patch("wiki_speedrun.choices._predict") as predict:
            result = policy.choose(NavigationEnvironment(graph, "Start", "Target"))
        predict.assert_not_called()
        self.assertEqual(result["selected_link_id"], "link_0")
        self.assertEqual(result["nodes"], [])

    def test_state_shortening_and_target_ancestry_labels(self):
        config = SimpleNamespace(node_max_choices=2, max_state_tokens=4,
                                 max_row_tokens=64)
        state = {"current_title": "Start", "target_title": "Target",
                 "current_excerpt": "abcdefgh", "history": ["Old", "Start"]}
        records = build_supervised_nodes(self.model, config, state,
                                         ["link_0", "link_1", "link_2"],
                                         ["A", "B", "Target"], "link_2", 19)
        self.assertTrue(records)
        self.assertTrue(all(record["label"] in record["candidate_ids"] for record in records))
        self.assertEqual(records[0]["label"], "link_2")
        self.assertGreater(records[0]["state_shortening"]["excerpt_chars_removed"], 0)
        self.assertIn("link_2", records[0]["tournament_trace"][0]["groups"][-1]["candidate_ids"])
        self.assertFalse(records[0]["tournament_trace"][0]["groups"][-1]["model_call"])

    def test_294_choices_are_covered_without_renumbering(self):
        moves = [f"Article_{index}" for index in range(294)]
        articles = set(moves + ["Start", "Target"])
        graph = WikiGraph(articles, {"Start": moves, **{article: [] for article in moves}}, {})
        config = SimpleNamespace(node_max_choices=64, max_state_tokens=32,
                                 max_row_tokens=4096, excerpt_chars=0)
        policy = TournamentPolicy(FakeModel(max_candidates=64), config)
        with patch("wiki_speedrun.choices._predict", side_effect=lambda model, request, ids, sample:
                   (ids[0], None, 0.0)):
            result = policy.choose(NavigationEnvironment(graph, "Start", "Target"))
        first_round = [node for node in result["nodes"] if node["round"] == 0]
        covered = [candidate_id for node in first_round
                   for candidate_id in node["candidate_ids"]]
        self.assertEqual(covered, [f"link_{index}" for index in range(294)])
        self.assertEqual(len(result["nodes"]), 6)

    def test_two_choices_that_cannot_fit_fail_instead_of_looping(self):
        graph = WikiGraph({"Start", "A", "B"}, {"Start": ["A", "B"]})
        policy = TournamentPolicy(FakeModel(max_candidates=1), self.config)
        with self.assertRaisesRegex(ValueError, "two candidate rows"):
            policy.choose(NavigationEnvironment(graph, "Start", "B"))


if __name__ == "__main__":
    unittest.main()
