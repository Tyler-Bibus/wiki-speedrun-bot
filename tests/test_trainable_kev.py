"""Focused tests for the lazy and request-conversion parts of the Kev adapter."""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from wiki_speedrun.models.trainable_kev import TrainableKev, _context_value, _hub_reference, _resolve_checkpoint


class TrainableKevTests(unittest.TestCase):
    def test_context_value_reads_objects_and_rejects_invalid_limits(self):
        self.assertEqual(_context_value({"max_state_tokens": 120}, "max_state_tokens", 1024), 120)
        self.assertEqual(_context_value(types.SimpleNamespace(max_row_tokens=900), "max_row_tokens", 4096), 900)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            _context_value({"max_state_tokens": 0}, "max_state_tokens", 1024)

    def test_hub_reference_preserves_optional_revision(self):
        self.assertEqual(_hub_reference("org/model@release"), ("org/model", "release"))
        self.assertEqual(_hub_reference("org/model"), ("org/model", None))

    def test_local_checkpoint_pointer_resolves_relative_target(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            run_directory = Path(temporary_directory)
            checkpoint_directory = run_directory / "step-12"
            checkpoint_directory.mkdir()
            (run_directory / "latest.json").write_text(
                json.dumps({"checkpoint": "step-12"}), encoding="utf-8"
            )
            self.assertEqual(_resolve_checkpoint(str(run_directory)), str(checkpoint_directory))
            pointer_path = run_directory / "latest.json"
            self.assertEqual(_resolve_checkpoint(str(pointer_path)), str(checkpoint_directory))

    def test_encode_preserves_order_and_uses_strict_context(self):
        kev_package = types.ModuleType("kev")
        kev_api = types.ModuleType("kev.api")

        class Request:
            @classmethod
            def model_validate(cls, request):
                return request

        def to_record(request):
            return {"state": request["state"], "questions": []}, [
                {"id": "next_link", "keys": list(request["questions"]["next_link"]["criteria"])}
            ]

        kev_api.SystemOneRequest = Request
        kev_api.to_record = to_record
        encode_arguments = []

        def encode(tokenizer, record, **kwargs):
            encode_arguments.append((tokenizer, record, kwargs))
            return {
                "ids": [1, 2, 3],
                "state_tokens": 2,
            }

        fake_model = types.SimpleNamespace(encode=encode)
        adapter = TrainableKev(
            tokenizer=object(),
            model=fake_model,
            metadata={},
            max_state_tokens=100,
            max_row_tokens=200,
        )
        request = {
            "state": "state",
            "questions": {"next_link": {"type": "choice", "criteria": {"link_8": "A", "link_2": "B"}}},
        }
        with patch.dict(sys.modules, {"kev": kev_package, "kev.api": kev_api}):
            encoded = adapter.encode(request)
        self.assertEqual(encoded["choice_ids"], ["link_8", "link_2"])
        self.assertEqual(encoded["token_counts"], {"state": 2, "row": 3})
        self.assertEqual(
            encode_arguments[0][2],
            {"max_state": 100, "max_branch": 200, "strict": True},
        )


if __name__ == "__main__":
    unittest.main()
