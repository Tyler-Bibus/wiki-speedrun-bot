"""Strict remote settings, separate from the training and legacy schemas."""
from dataclasses import dataclass
from pathlib import Path
import math
import tomllib


@dataclass
class RemoteConfig:
    model: str = 'typesafe/jev-1.14'
    protocol: str = 'matched_kev_tournament'
    data_dir: str = 'data'
    run_dir: str = 'runs'
    credentials_file: str = '.env.openrouter'
    key_variable: str = 'OPENROUTER_API_KEY'
    dev_tasks: str = 'data/splits/training/dev_tasks.json'
    test_tasks: str = 'data/splits/training/test_tasks.json'
    train_tasks: str = 'data/splits/training/train_tasks.json'
    split_manifest: str = 'data/splits/training/manifest.json'
    tokenizer: str = 'checkpoints/20261022T025733.517853Z-sft/update-001700'
    max_steps: int = 12
    excerpt_chars: int = 1500
    node_max_choices: int = 64
    max_state_tokens: int = 1024
    max_row_tokens: int = 4096
    min_request_interval: float = 0
    timeout: float = 120
    retries: int = 2
    max_requests: int = 15000
    max_cost: float = 5.0
    max_completion_tokens: int = 2048
    temperature: float = 0
    reasoning_effort: str = 'low'
    provider: str = ''
    structured_output: str = 'json_schema'
    prompt_version: str = 'navigation-v1'
    schema_version: int = 1

    def __post_init__(self):
        if self.protocol not in ('matched_kev_tournament', 'llm_full_list_json', 'llm_full_list_json_object'):
            raise ValueError('Unknown remote protocol')
        if not self.model or self.structured_output not in ('json_schema', 'json_object'):
            raise ValueError('Explicit model and structured-output mode required')
        if self.protocol.startswith('llm_'):
            expected = 'llm_full_list_json' if self.structured_output == 'json_schema' else 'llm_full_list_json_object'
            if self.protocol != expected:
                raise ValueError('JSON-object mode must have its own protocol label')
        for name in ('max_steps', 'max_state_tokens', 'max_row_tokens', 'max_requests', 'max_completion_tokens'):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(name + ' must be a positive integer')
        if not 2 <= self.node_max_choices <= 255 or self.excerpt_chars < 0 or self.retries not in (0, 1, 2):
            raise ValueError('Invalid remote context or retry limits')
        if not all(math.isfinite(x) for x in (self.max_cost, self.timeout, self.temperature, self.min_request_interval)) or self.max_cost <= 0 or self.timeout <= 0 or self.temperature < 0 or self.min_request_interval < 0:
            raise ValueError('Invalid remote cost, timeout, or temperature')


def load_remote_config(path):
    return RemoteConfig(**tomllib.loads(Path(path).read_text()))
