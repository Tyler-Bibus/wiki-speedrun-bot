"""Standard-library client for Kev's System One choice endpoint."""

from __future__ import annotations

import json
import math
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class KevClient:
    """Call a Kev server's ``POST /v1/systemone`` endpoint."""

    def __init__(self, base_url: str = "http://127.0.0.1:8008", timeout: float = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def score(self, request: dict) -> dict[str, Any]:
        """Submit a System One request and return its next_link choice answer.

        The request must contain ``questions.next_link`` as a choice question with
        criteria keyed by the allowed link labels.
        """
        try:
            questions = request["questions"]
            question = questions["next_link"]
            criteria = question["criteria"]
            if question.get("type") != "choice" or not isinstance(criteria, dict):
                raise ValueError("questions.next_link must be a choice question with a criteria object")
            allowed = set(criteria)
            if not allowed:
                raise ValueError("questions.next_link.criteria must contain at least one choice")
        except (KeyError, TypeError) as exc:
            raise ValueError("request must include questions.next_link.criteria") from exc

        payload = json.dumps(request).encode("utf-8")
        http_request = Request(
            f"{self.base_url}/v1/systemone",
            data=payload,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urlopen(http_request, timeout=self.timeout) as response:
                raw = response.read()
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"Kev returned HTTP {exc.code}{suffix}") from exc
        except (URLError, TimeoutError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise RuntimeError(f"Could not reach Kev at {self.base_url}: {reason}") from exc

        try:
            body = json.loads(raw.decode("utf-8"))
            answer = body["answers"]["next_link"]
            choice = answer["choice"]
            probabilities = answer["probabilities"]
            confidence = answer.get("confidence")
            if not isinstance(choice, str):
                raise TypeError("choice is not a string")
            if not isinstance(probabilities, dict) or any(
                not isinstance(key, str) or not isinstance(value, (int, float))
                for key, value in probabilities.items()
            ):
                raise TypeError("probabilities is not a string-to-number object")
            if confidence is not None and not isinstance(confidence, (int, float)):
                raise TypeError("confidence is not a number or null")
            if any(not math.isfinite(float(value)) or not 0 <= value <= 1 for value in probabilities.values()):
                raise ValueError("probabilities contain a non-finite or out-of-range value")
            if confidence is not None and (not math.isfinite(float(confidence)) or not 0 <= confidence <= 1):
                raise ValueError("confidence is non-finite or outside [0, 1]")
        except (UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Kev returned an invalid System One response: {exc}") from exc

        if choice not in allowed:
            raise RuntimeError(
                f"Kev returned choice {choice!r}, which is not one of the submitted criteria"
            )
        return {
            "choice": choice,
            "probabilities": {key: float(value) for key, value in probabilities.items()},
            "confidence": float(confidence) if confidence is not None else None,
        }

    def smoke_test(self) -> dict[str, Any]:
        """Ask a small known-fact question to check the endpoint end to end."""
        request = {
            "state": "Earth is the third planet from the Sun.",
            "model": "kev-latest",
            "questions": {
                "next_link": {
                    "type": "choice",
                    "instructions": "Which planet is described?",
                    "criteria": {"earth": "The third planet from the Sun", "mars": "The fourth planet from the Sun"},
                }
            },
        }
        result = self.score(request)
        if result["choice"] != "earth":
            raise RuntimeError(f"Kev smoke test expected 'earth', got {result['choice']!r}")
        return result

