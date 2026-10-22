"""Small experiment settings shared by the command-line commands."""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass
class Config:
    data_dir: str = "data"
    run_dir: str = "runs"
    base_url: str = "http://127.0.0.1:8008"
    checkpoint: str = "jaredpalmer/kev-0.8b@v1.0"
    max_steps: int = 12
    excerpt_chars: int = 1500
    max_options: int = 255
    seed: int = 42
    timeout: float = 120.0

    def __post_init__(self):
        if self.max_steps < 1 or self.excerpt_chars < 0:
            raise ValueError("max_steps must be positive and excerpt_chars nonnegative")
        if not 1 <= self.max_options <= 255:
            raise ValueError("max_options must be between 1 and 255")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")


def load_config(path: Path) -> Config:
    with path.open("rb") as handle:
        return Config(**tomllib.load(handle))
