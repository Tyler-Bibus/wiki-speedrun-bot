"""A shared reported-cost ceiling across every run and probe in one suite."""
import math
from pathlib import Path
from threading import RLock

from .models.openrouter import RemoteError, finite
from .training.checkpoints import write_json_atomic


class SuiteBudget:
    def __init__(self, output, limit):
        from .remote import read_rows
        if isinstance(limit, bool) or not math.isfinite(limit) or limit <= 0:
            raise ValueError('--max-total-cost must be finite and positive')
        self.output = Path(output)
        self._lock = RLock()
        self.limit = limit
        self.known_cost = 0.0
        self.attempts = 0
        self.unknown_cost_attempts = 0
        self.unpriced_responses = 0
        for path in self.output.rglob('requests.jsonl'):
            for row in read_rows(path):
                self._count(row)
        self.save()

    def _count(self, row):
        self.attempts += 1
        response = row.get('response')
        usage = response.get('usage') if isinstance(response, dict) else None
        cost = usage.get('cost') if isinstance(usage, dict) else None
        if finite(cost):
            self.known_cost += cost
        else:
            self.unknown_cost_attempts += 1
            if isinstance(response, dict):
                self.unpriced_responses += 1

    def check(self):
        with self._lock:
            if self.unpriced_responses:
                raise RemoteError('suite_budget', 'A response lacks cost; suite stops until spend can be reconciled')
            if self.known_cost >= self.limit:
                raise RemoteError('suite_budget', 'Shared benchmark reported-cost limit reached')

    def observe(self, row):
        with self._lock:
            self._count(row)
            self.save()

    def save(self):
        with self._lock:
            self._save()

    def _save(self):
        write_json_atomic(self.output / 'budget.json', {
            'max_total_cost': self.limit, 'known_cost': self.known_cost,
            'remaining_reported_budget': max(0, self.limit - self.known_cost),
            'http_attempts': self.attempts, 'unknown_cost_attempts': self.unknown_cost_attempts,
            'unpriced_responses': self.unpriced_responses,
            'note': 'Includes all suite probes, retries, and abandoned episode attempts. '
                    'Requests already in flight can overshoot; unknown charges are not assumed zero.',
        })
