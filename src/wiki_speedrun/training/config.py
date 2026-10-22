"""Explicit, validated settings for the optional training commands."""

from dataclasses import asdict, dataclass, field, fields
import math
from pathlib import Path
import tomllib
from typing import Any


@dataclass
class ModelConfig:
    checkpoint: str = "jaredpalmer/kev-0.8b@v1.0"
    kev_revision: str = "6b719c3c3f367295f6ef336f4f751cf5ff970abc"
    device: str = "cuda"
    weights_dtype: str = "fp32"
    autocast_dtype: str = "bf16"
    temperature: float = 1.0


@dataclass
class ChoiceConfig:
    strategy: str = "tournament"
    node_max_choices: int = 64
    max_state_tokens: int = 1024
    max_row_tokens: int = 4096
    excerpt_chars: int = 1500
    grouping: str = "source_order"


@dataclass
class DataConfig:
    max_paths: int = 2000
    split_manifest: str = "data/splits/training/manifest.json"
    sft_train: str = "data/training/sft_train.jsonl"
    sft_dev: str = "data/training/sft_dev.jsonl"
    rl_train_tasks: str = "data/splits/training/train_tasks.json"
    dev_tasks: str = "data/splits/training/dev_tasks.json"
    test_tasks: str = "data/splits/training/test_tasks.json"


@dataclass
class SftConfig:
    epochs: int = 1
    max_updates: int = 200
    learning_rate: float = 2e-5
    microbatch_size: int = 1
    gradient_accumulation: int = 8


@dataclass
class RlConfig:
    max_steps: int = 12
    updates: int = 20
    task_pairs_per_update: int = 2
    episodes_per_pair: int = 2
    learning_rate: float = 5e-6
    gamma: float = 1.0
    entropy_coefficient: float = 0.0
    max_grad_norm: float = 1.0


@dataclass
class RewardConfig:
    success: float = 1.0
    failure: float = -1.0
    step_cost: float = 0.02


@dataclass
class SaveConfig:
    checkpoint_every_updates: int = 50
    keep_last_checkpoints: int = 2
    evaluate_every_updates: int = 100
    evaluation_max_tasks: int = 16


@dataclass
class TrainingConfig:
    schema_version: int = 1
    seed: int = 42
    data_dir: str = "data"
    run_dir: str = "runs"
    model: ModelConfig = field(default_factory=ModelConfig)
    choices: ChoiceConfig = field(default_factory=ChoiceConfig)
    data: DataConfig = field(default_factory=DataConfig)
    sft: SftConfig = field(default_factory=SftConfig)
    rl: RlConfig = field(default_factory=RlConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    save: SaveConfig = field(default_factory=SaveConfig)

    def __post_init__(self) -> None:
        if type(self.schema_version) is not int or self.schema_version != 1:
            raise ValueError("Training schema_version must be 1")
        integer_settings = {
            "seed": self.seed,
            "choices.node_max_choices": self.choices.node_max_choices,
            "choices.max_state_tokens": self.choices.max_state_tokens,
            "choices.max_row_tokens": self.choices.max_row_tokens,
            "choices.excerpt_chars": self.choices.excerpt_chars,
            "data.max_paths": self.data.max_paths,
            "sft.epochs": self.sft.epochs,
            "sft.max_updates": self.sft.max_updates,
            "sft.microbatch_size": self.sft.microbatch_size,
            "sft.gradient_accumulation": self.sft.gradient_accumulation,
            "rl.max_steps": self.rl.max_steps,
            "rl.updates": self.rl.updates,
            "rl.task_pairs_per_update": self.rl.task_pairs_per_update,
            "rl.episodes_per_pair": self.rl.episodes_per_pair,
            "save.checkpoint_every_updates": self.save.checkpoint_every_updates,
            "save.keep_last_checkpoints": self.save.keep_last_checkpoints,
            "save.evaluate_every_updates": self.save.evaluate_every_updates,
            "save.evaluation_max_tasks": self.save.evaluation_max_tasks,
        }
        for name, value in integer_settings.items():
            if type(value) is not int:
                raise ValueError(f"{name} must be an integer")
            minimum = 0 if name in {"seed", "choices.excerpt_chars"} else 1
            if value < minimum:
                raise ValueError(f"{name} must be at least {minimum}")
        if self.seed > 2**32 - 1:
            raise ValueError("seed must fit in 32 bits for NumPy reproducibility")
        if not 2 <= self.choices.node_max_choices <= 255:
            raise ValueError("choices.node_max_choices must be between 2 and 255")
        if self.choices.max_row_tokens <= self.choices.max_state_tokens:
            raise ValueError("choices.max_row_tokens must exceed max_state_tokens")
        if self.choices.strategy != "tournament" or self.choices.grouping != "source_order":
            raise ValueError("Only tournament strategy with source_order grouping is supported")
        if self.sft.microbatch_size != 1:
            raise ValueError("Version 1 uses sft.microbatch_size = 1; use gradient_accumulation")
        if self.rl.episodes_per_pair < 2:
            raise ValueError("rl.episodes_per_pair must be at least 2 for the baseline")
        positive_settings = {
            "sft.learning_rate": self.sft.learning_rate,
            "rl.learning_rate": self.rl.learning_rate,
            "rl.max_grad_norm": self.rl.max_grad_norm,
        }
        for name, value in positive_settings.items():
            _finite_number(name, value)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        for name, value in {
            "reward.success": self.reward.success,
            "reward.failure": self.reward.failure,
            "reward.step_cost": self.reward.step_cost,
            "rl.entropy_coefficient": self.rl.entropy_coefficient,
            "rl.gamma": self.rl.gamma,
            "model.temperature": self.model.temperature,
        }.items():
            _finite_number(name, value)
        if self.reward.step_cost <= 0 or self.rl.entropy_coefficient < 0:
            raise ValueError("step_cost must be positive and entropy_coefficient nonnegative")
        if self.reward.success <= 0 or self.reward.failure >= 0:
            raise ValueError("Success must have positive reward and failure negative reward")
        if self.reward.step_cost * self.rl.max_steps >= self.reward.success - self.reward.failure:
            raise ValueError("Step cost is too large: a success must beat every failure")
        if self.rl.gamma != 1.0 or self.model.temperature != 1.0:
            raise ValueError("Version 1 uses rl.gamma = 1.0 and model.temperature = 1.0")
        if self.model.weights_dtype != "fp32" or self.model.autocast_dtype not in {"bf16", "fp32"}:
            raise ValueError("Use fp32 weights and bf16 or fp32 autocast")
        data_paths = {name: value for name, value in asdict(self.data).items()
                      if name != "max_paths"}
        for name, value in {"data_dir": self.data_dir, "run_dir": self.run_dir,
                            "model.checkpoint": self.model.checkpoint,
                            "model.device": self.model.device,
                            "model.kev_revision": self.model.kev_revision,
                            **{f"data.{key}": value for key, value in data_paths.items()}}.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        if self.model.device not in {"cpu", "cuda"}:
            prefix, separator, index = self.model.device.partition(":")
            if prefix != "cuda" or separator != ":" or not index.isdecimal():
                raise ValueError("model.device must be cpu, cuda, or cuda:<index>")
        if self.model.kev_revision != ModelConfig().kev_revision:
            raise ValueError("model.kev_revision must match the pinned Kev version in pyproject.toml")
        paths = list(data_paths.values())
        if len({Path(path).resolve() for path in paths}) != len(paths):
            raise ValueError("Training data output paths must be distinct")


def _finite_number(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


def load_training_config(path: Path) -> TrainingConfig:
    """Load only known sections and keys, so typos never silently change a run."""
    with Path(path).open("rb") as stream:
        values = tomllib.load(stream)
    sections = {
        "model": ModelConfig,
        "choices": ChoiceConfig,
        "data": DataConfig,
        "sft": SftConfig,
        "rl": RlConfig,
        "reward": RewardConfig,
        "save": SaveConfig,
    }
    known = {item.name for item in fields(TrainingConfig)}
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    for name, cls in sections.items():
        section = values.get(name, {})
        if not isinstance(section, dict):
            raise ValueError(f"{name} must be a TOML section")
        unknown = set(section) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown {name} settings: {sorted(unknown)}")
        values[name] = cls(**section)
    return TrainingConfig(**values)
