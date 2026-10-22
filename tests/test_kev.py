"""Tests for the local Kev HTTP client."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from wiki_speedrun.models import KevClient


class _Handler(BaseHTTPRequestHandler):
    response_body = {}
    status = 200
    received = None

    def do_POST(self):
        type(self).received = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        body = json.dumps(type(self).response_body).encode()
        self.send_response(type(self).status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class KevClientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        _Handler.status = 200
        _Handler.response_body = {
            "answers": {"next_link": {"type": "choice", "choice": "Paris", "probabilities": {"Paris": 0.8, "London": 0.2}, "confidence": 0.6}}
        }
        _Handler.received = None
        self.request = {"state": "France's capital is Paris.", "model": "kev-latest", "questions": {"next_link": {"type": "choice", "instructions": "Choose the capital", "criteria": {"Paris": "Capital of France", "London": "Capital of UK"}}}}
        self.client = KevClient(self.base_url, timeout=2)

    def test_posts_and_returns_choice_distribution(self):
        result = self.client.score(self.request)
        self.assertEqual(result, {"choice": "Paris", "probabilities": {"Paris": 0.8, "London": 0.2}, "confidence": 0.6})
        self.assertEqual(_Handler.received, self.request)

    def test_rejects_choice_outside_submitted_criteria(self):
        _Handler.response_body["answers"]["next_link"]["choice"] = "Rome"
        with self.assertRaisesRegex(RuntimeError, "not one of the submitted criteria"):
            self.client.score(self.request)

    def test_rejects_malformed_response(self):
        _Handler.response_body = {"answers": {}}
        with self.assertRaisesRegex(RuntimeError, "invalid System One response"):
            self.client.score(self.request)

    def test_reports_http_failure(self):
        _Handler.status = 503
        with self.assertRaisesRegex(RuntimeError, "HTTP 503"):
            self.client.score(self.request)


if __name__ == "__main__":
    unittest.main()
