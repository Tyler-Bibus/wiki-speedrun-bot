"""Authenticated fixed-origin OpenRouter transports; no optional dependencies."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import json
import math
import socket
import time
import urllib.error
import urllib.request


class RemoteError(RuntimeError):
    def __init__(self, kind, message):
        self.kind = kind
        super().__init__(message)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON constant')
    def number(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError('nonfinite JSON number')
        return parsed
    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=number)


def finite(value, low=0, high=float('inf')):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and low <= value <= high)


def provider_error(detail, message):
    """Classify explicit provider status codes inside HTTP-success envelopes."""
    code = detail.get('code') if isinstance(detail, dict) else None
    for status in (429, 500, 502, 503, 504, 529):
        if code == status or code == str(status):
            kind = 'rate_limit' if status == 429 else 'provider_unavailable'
            return RemoteError(kind, f'{message} (HTTP {status})')
    return RemoteError('api', message)


def metadata(data):
    if not isinstance(data, dict) or not all(isinstance(data.get(k), str) and data[k]
                                           for k in ('id', 'model', 'provider')):
        raise RemoteError('schema', 'OpenRouter response lacks request/model/provider identity')
    usage = data.get('usage')
    if usage is not None:
        if not isinstance(usage, dict):
            raise RemoteError('schema', 'Invalid OpenRouter usage')
        for key in ('cost', 'input_tokens', 'output_tokens', 'prompt_tokens', 'completion_tokens', 'total_tokens'):
            value = usage.get(key)
            if value is not None and (not finite(value) or (key != 'cost' and not isinstance(value, int))):
                raise RemoteError('schema', 'Invalid OpenRouter usage value')
        for field in ('prompt_tokens_details', 'completion_tokens_details', 'input_tokens_details', 'output_tokens_details'):
            details = usage.get(field)
            if details is not None and (not isinstance(details, dict) or any(
                    value is not None and name.endswith('_tokens') and
                    (not isinstance(value, int) or not finite(value))
                    for name, value in details.items())):
                raise RemoteError('schema', 'Invalid OpenRouter detailed token usage')
    return {k: data.get(k) for k in ('id', 'model', 'provider', 'usage')}


class OpenRouterClient:
    endpoint = ''

    def __init__(self, key, model, timeout=120, retries=2, *, transport=None,
                 on_attempt=None, before_attempt=None, sleep=time.sleep, min_request_interval=0):
        if retries not in (0, 1, 2) or timeout <= 0:
            raise ValueError('timeout must be positive and retries in 0..2')
        self._key = key
        self.model = model
        self.timeout = timeout
        self.retries = retries
        self._transport = transport or urllib.request.build_opener(NoRedirect()).open
        self.on_attempt = on_attempt
        self.before_attempt = before_attempt
        self.sleep = sleep
        self.resolved_model = None
        self.context = {}
        self.min_request_interval = min_request_interval
        self._last_submission = None

    def redact(self, value):
        if isinstance(value, str):
            return value.replace(self._key, '[REDACTED]')
        if isinstance(value, list):
            return [self.redact(v) for v in value]
        if isinstance(value, dict):
            return {self.redact(k): self.redact(v) for k, v in value.items()}
        return value

    def _post(self, payload, validate):
        started = time.perf_counter()
        for attempt in range(1, self.retries + 2):
            if self.before_attempt:
                self.before_attempt()
            throttle_wait = (max(0, self.min_request_interval - (time.monotonic() - self._last_submission))
                             if self._last_submission is not None else 0)
            if throttle_wait:
                self.sleep(throttle_wait)
            self._last_submission = time.monotonic()
            record = {**self.context, 'throttle_wait_seconds': throttle_wait, 'attempt': attempt, 'requested_model': self.model,
                      'endpoint': self.endpoint, 'request': payload}
            retry_delay = None
            error = None
            call_started = time.perf_counter()
            try:
                request = urllib.request.Request('https://openrouter.ai' + self.endpoint,
                    data=json.dumps(payload, ensure_ascii=False, allow_nan=False).encode(),
                    headers={'Authorization': 'Bearer ' + self._key, 'Content-Type': 'application/json'},
                    method='POST')
                with self._transport(request, timeout=self.timeout) as response:
                    raw = response.read()
                try:
                    data = strict_json(raw)
                except (ValueError, UnicodeError):
                    raise RemoteError('schema', 'OpenRouter returned an invalid JSON envelope') from None
                if not isinstance(data, dict) or 'error' in data:
                    raise provider_error(data.get('error') if isinstance(data, dict) else None,
                                         'OpenRouter returned an API error envelope')
                # Save billed usage even when subsequent contract validation fails.
                record['response'] = data
                info = metadata(data)
                if self.resolved_model is not None and self.resolved_model != info['model']:
                    raise RemoteError('model_changed', 'OpenRouter resolved model changed during the run')
                self.resolved_model = info['model']
                result = validate(data)
                record['result'] = result
                result.update(info)
                result['attempts'] = attempt
                result['latency_ms'] = (time.perf_counter() - started) * 1000
            except urllib.error.HTTPError as exc:
                # Provider bodies and URLs are untrusted and are never included in errors.
                code = exc.code
                detail = exc.read(65536).decode('utf-8', errors='replace').lower()
                kind = ('context_limit' if code in (400, 413, 422) and
                        any(s in detail for s in ('context', 'too long', 'token limit')) else
                        'unsupported_parameters' if code in (400, 404, 422) and
                        any(s in detail for s in ('requested parameters', 'structured output', 'json_schema')) else
                        'daily_quota' if code == 429 and any(s in detail for s in ('daily', 'per-day', 'per day')) else
                        'rate_limit' if code == 429 else
                        'auth' if code in (401, 402, 403) else 'http')
                error = RemoteError(kind, f'OpenRouter HTTP {code} ({kind})')
                record['http_status'] = code
                if code in (429, 500, 502, 503, 504, 529) and kind != 'daily_quota' and attempt <= self.retries:
                    retry_delay = 30 if code == 429 else 2 ** (attempt - 1)
                    retry_after = exc.headers.get('Retry-After') if exc.headers else None
                    if retry_after:
                        try:
                            retry_delay = float(retry_after)
                        except ValueError:
                            try:
                                retry_delay = (parsedate_to_datetime(retry_after) - datetime.now(timezone.utc)).total_seconds()
                            except (ValueError, TypeError, OverflowError):
                                pass
                    retry_delay = min(30, max(0, retry_delay))
            except (urllib.error.URLError, TimeoutError, socket.timeout, OSError):
                error = RemoteError('network', 'OpenRouter networking failure; ambiguous submission is not retried')
            except RemoteError as exc:
                error = exc
                if exc.kind in ('provider_unavailable', 'rate_limit') and attempt <= self.retries:
                    retry_delay = 30 if exc.kind == 'rate_limit' else 2 ** (attempt - 1)
            finally:
                record['latency_ms'] = (time.perf_counter() - call_started) * 1000
                record['elapsed_ms'] = (time.perf_counter() - started) * 1000
                if error:
                    record['error'] = {'kind': error.kind, 'message': str(error)}
                if retry_delay is not None:
                    record['retry_delay_seconds'] = retry_delay
                if self.on_attempt:
                    self.on_attempt(self.redact(record))
            if retry_delay is not None:
                self.sleep(retry_delay)
                continue
            if error:
                raise error from None
            return result


class OpenRouterDecisionClient(OpenRouterClient):
    endpoint = '/api/alpha/decisions'

    def score(self, request):
        payload = copy.deepcopy(request)
        payload['model'] = self.model
        criteria = payload['questions']['next_link']['criteria']
        if not 1 <= len(criteria) <= 255:
            raise ValueError('decision criteria must have 1..255 options')

        def validate(data):
            answers = data.get('answers')
            answer = answers.get('next_link') if isinstance(answers, dict) else None
            if (not isinstance(answer, dict) or answer.get('type') != 'choice'
                    or not isinstance(answer.get('choice'), str) or answer['choice'] not in criteria):
                raise RemoteError('schema', 'Invalid next_link Choice answer')
            probs = answer.get('probabilities')
            if (not isinstance(probs, dict) or set(probs) != set(criteria)
                    or not all(finite(p, 0, 1) for p in probs.values())):
                raise RemoteError('schema', 'Invalid or incomplete Choice distribution')
            # Decisions API probabilities are rounded to two decimals. Check
            # whether a unit-sum distribution fits their rounding intervals;
            # rounding error accumulates across options. Keep the raw values
            # and the provider's selected choice unchanged.
            lower = math.fsum(max(0, p - 0.005) for p in probs.values())
            upper = math.fsum(min(1, p + 0.005) for p in probs.values())
            if math.fsum(probs.values()) <= 0 or lower > 1 + 1e-12 or upper < 1 - 1e-12:
                raise RemoteError('schema', 'Invalid Choice distribution beyond two-decimal rounding')
            confidence = answer.get('confidence')
            if confidence is not None and not finite(confidence, 0, 1):
                raise RemoteError('schema', 'Invalid Choice confidence')
            return {'choice': answer['choice'], 'probabilities': probs, 'confidence': confidence}
        return self._post(payload, validate)

    def smoke_test(self):
        return self.score({'state': 'Select the planet humans live on.', 'questions': {
            'next_link': {'type': 'choice', 'instructions': 'Choose Earth.',
                          'criteria': {'earth': 'Earth', 'mars': 'Mars'}}}})


class OpenRouterChatClient(OpenRouterClient):
    endpoint = '/api/v1/chat/completions'

    def complete(self, request):
        payload = copy.deepcopy(request)
        payload['model'] = self.model

        def validate(data):
            choices = data.get('choices')
            if (not isinstance(choices, list) or len(choices) != 1
                    or not isinstance(choices[0], dict) or not isinstance(choices[0].get('message'), dict)
                    or not isinstance(choices[0].get('finish_reason'), str)):
                raise RemoteError('schema', 'Invalid OpenRouter chat response envelope')
            choice = choices[0]
            if choice['finish_reason'] == 'error' or 'error' in choice:
                detail = choice.get('error')
                if detail is None:
                    detail = choice['message'].get('error')
                message = 'OpenRouter provider failed during completion'
                # Some providers return a valid envelope with no error details
                # after generation fails. Retry this explicit generation failure,
                # while leaving unrecognized structured API errors fatal.
                if choice['finish_reason'] == 'error' and detail is None:
                    raise RemoteError('provider_unavailable', message)
                raise provider_error(detail, message)
            return {'content': choice['message'].get('content'),
                    'refusal': choice['message'].get('refusal'),
                    'finish_reason': choice['finish_reason']}
        return self._post(payload, validate)
