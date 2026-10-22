"""Differentiable in-process adapter for the pinned Kev decision model.

The optional training dependency is imported only when this adapter is loaded or
used, so the HTTP baseline remains usable in a standard-library environment.
"""

from __future__ import annotations

import os
import json
import importlib.metadata
from pathlib import Path
from typing import Any


KEV_REVISION = "6b719c3c3f367295f6ef336f4f751cf5ff970abc"
CHECKPOINT_ID = "jaredpalmer/kev-0.8b@v1.0"


def _context_value(context: object | None, name: str, default: int) -> int:
    if context is None:
        return default
    if isinstance(context, dict):
        value = context.get(name, default)
    else:
        value = getattr(context, name, default)
    if not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _lora_targets(adapter_config: dict[str, Any]) -> str:
    target_modules = set(adapter_config["target_modules"])
    all_targets = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    attention_targets = {"q_proj", "k_proj", "v_proj", "o_proj"}
    hybrid_targets = {"in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"}
    targets_by_name = [
        ("all", all_targets | hybrid_targets),
        ("attn", attention_targets | hybrid_targets),
        ("dense", all_targets),
        ("attn", attention_targets),
        ("qv", {"q_proj", "v_proj"}),
    ]
    for name, expected in targets_by_name:
        if target_modules == expected:
            return name
    raise ValueError(f"unsupported Kev LoRA target modules: {sorted(target_modules)}")


class TrainableKev:
    """Load a Kev checkpoint with its existing LoRA adapter and pointer head trainable."""

    def __init__(
        self,
        tokenizer: object,
        model: object,
        metadata: dict[str, Any],
        max_state_tokens: int,
        max_row_tokens: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.model = model
        self.metadata = metadata
        self.max_state_tokens = max_state_tokens
        self.max_row_tokens = max_row_tokens

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str,
        device: str = "cuda",
        context: object | None = None,
    ) -> TrainableKev:
        """Build Kev in fp32 and warm-start all adapter and pointer-head weights."""
        repository_root = Path(__file__).resolve().parents[3]
        os.environ.setdefault("HF_HOME", str(repository_root / ".cache" / "huggingface"))
        os.environ.setdefault("TRITON_CACHE_DIR", str(repository_root / ".cache" / "triton"))
        import torch
        from huggingface_hub import HfApi
        from kev.checkpoint import Checkpoint, Meta
        from kev.model import DecisionModel, load_tokenizer

        installed_revision = _installed_kev_revision()
        if installed_revision != KEV_REVISION:
            raise RuntimeError(
                f"Expected Kev source revision {KEV_REVISION}, got {installed_revision!r}"
            )
        checkpoint_ref = _resolve_checkpoint(checkpoint or CHECKPOINT_ID)
        source_ref = checkpoint_ref
        checkpoint_revision = None
        if not Path(checkpoint_ref).is_dir():
            repository, revision = _hub_reference(checkpoint_ref)
            checkpoint_revision = HfApi().model_info(repository, revision=revision).sha
            source_ref = f"{repository}@{checkpoint_revision}"
        source = Checkpoint(source_ref)
        source_meta = source.meta
        if source.full:
            raise ValueError("TrainableKev supports LoRA checkpoints with an existing Kev adapter")
        base_revision = source_meta.base_revision or HfApi().model_info(source_meta.base).sha
        tokenizer = load_tokenizer(source_meta.base, revision=base_revision)
        adapter_config = source.adapter_config()
        targets = _lora_targets(adapter_config)
        model = DecisionModel(
            source_meta.base,
            tokenizer,
            device,
            lora=source_meta.lora,
            revision=base_revision,
            lora_targets=targets,
            head_dim=source_meta.head_dim,
            option_isolation=source_meta.option_isolation,
            special_embeddings=source_meta.special_embeddings,
            dtype=torch.float32,
            attn="sdpa" if device.startswith("cuda") else "eager",
        )
        run_meta = Meta(
            base=source_meta.base,
            base_revision=base_revision,
            lora=source_meta.lora,
            head_dim=source_meta.head_dim,
            option_isolation=source_meta.option_isolation,
            special_embeddings=source_meta.special_embeddings,
            weights_dtype="fp32",
            temperature=1.0,
        )
        warm_start = source.warm_start(model, run_meta)
        model.head.temperature = 1.0
        _assert_trainable_parameters(model)

        max_state_tokens = _context_value(context, "max_state_tokens", 1024)
        max_row_tokens = _context_value(context, "max_row_tokens", 4096)
        return cls(
            tokenizer,
            model,
            {
                "checkpoint": checkpoint_ref,
                "resolved_checkpoint": source.path,
                "checkpoint_revision": checkpoint_revision or source.meta.extra.get("provenance", {}).get("checkpoint_revision"),
                "kev_revision": KEV_REVISION,
                "base": source_meta.base,
                "base_revision": base_revision,
                "temperature": 1.0,
                "weights_dtype": "fp32",
                "lora": source_meta.lora,
                "head_dim": source_meta.head_dim,
                "option_isolation": source_meta.option_isolation,
                "special_embeddings": source_meta.special_embeddings,
                "warm_start": warm_start,
            },
            max_state_tokens,
            max_row_tokens,
        )

    def encode(self, request: dict[str, Any]) -> dict[str, Any]:
        """Strictly encode a System One request and retain its original choice-key order."""
        from kev.api import SystemOneRequest, to_record

        parsed_request = SystemOneRequest.model_validate(request)
        record, question_metadata = to_record(parsed_request)
        if len(question_metadata) != 1 or question_metadata[0]["id"] != "next_link":
            raise ValueError("a Kev navigation request must contain only questions.next_link")
        choice_ids = question_metadata[0]["keys"]
        if not choice_ids:
            raise ValueError("questions.next_link must contain at least one choice")
        encoded = self.model.encode(
            self.tokenizer,
            record,
            max_state=self.max_state_tokens,
            max_branch=self.max_row_tokens,
            strict=True,
        )
        return {
            "encoded": encoded,
            "choice_ids": list(choice_ids),
            "token_counts": {
                "state": encoded["state_tokens"],
                "row": len(encoded["ids"]),
            },
        }

    def logits(self, encoded: dict[str, Any]) -> Any:
        """Return raw differentiable float32 logits in the supplied choice order."""
        import torch

        scores = self.model.forward(encoded["encoded"])[0]
        return scores.to(dtype=torch.float32)

    def fits(self, request: dict[str, Any]) -> bool:
        """Return whether the request fits, catching only Kev's context overflow."""
        from kev.model import ContextOverflow

        try:
            self.encode(request)
        except ContextOverflow:
            return False
        return True

    def state_token_count(self, state: object) -> int:
        """Count Kev state tokens, including the state delimiter."""
        from kev.api import render
        from kev.model import user_tokens

        return len(user_tokens(self.tokenizer, render(state))) + 1

    def token_counts(self, request: dict[str, Any]) -> dict[str, int]:
        """Return state and full-row token counts for logs and trace records."""
        counts = self.encode(request)["token_counts"]
        return {"state": int(counts["state"]), "row": int(counts["row"])}

    def option_token_count(self, choice_id: str, title: str) -> int:
        """Count one encoded option and its Kev open/close tokens."""
        from kev.api import option_text
        from kev.model import user_tokens

        from ..data.load import display_title

        text = option_text(choice_id, display_title(title))
        return len(user_tokens(self.tokenizer, text)) + 2

    def instruction_token_count(self, instructions: str) -> int:
        """Count instruction tokens plus Kev's question and decision markers."""
        from kev.model import user_tokens

        return len(user_tokens(self.tokenizer, instructions)) + 2

    def parameters(self):
        return self.model.parameters()

    @property
    def training(self) -> bool:
        return self.model.training

    def train(self, mode: bool = True) -> TrainableKev:
        self.model.train(mode)
        return self

    def eval(self) -> TrainableKev:
        self.model.eval()
        return self

    def save_checkpoint(self, directory: Path) -> None:
        """Write an upstream-compatible LoRA adapter, tokenizer, and head.pt metadata."""
        from kev.checkpoint import Meta, write_meta

        output_directory = Path(directory)
        output_directory.mkdir(parents=True, exist_ok=True)
        self.model.lm.save_pretrained(output_directory, safe_serialization=True)
        self.tokenizer.save_pretrained(output_directory)
        meta = Meta(
            base=self.metadata["base"],
            base_revision=self.metadata["base_revision"],
            lora=self.metadata["lora"],
            head_dim=self.metadata["head_dim"],
            option_isolation=self.metadata["option_isolation"],
            special_embeddings=self.metadata["special_embeddings"],
            weights_dtype="fp32",
            temperature=1.0,
            head=self.model.head.state_dict(),
            extra={"provenance": self.metadata},
        )
        write_meta(output_directory, meta)


def _hub_reference(checkpoint: str) -> tuple[str, str | None]:
    repository, separator, revision = checkpoint.partition("@")
    if not separator:
        revision = None
    return repository, revision


def _resolve_checkpoint(checkpoint: str) -> str:
    """Resolve common run-directory pointers while leaving Hub IDs untouched."""
    checkpoint_path = Path(checkpoint)
    if checkpoint_path.is_file() and checkpoint_path.name in {"best.json", "latest.json"}:
        return _resolve_pointer(checkpoint_path)
    if not checkpoint_path.is_dir():
        return checkpoint
    for pointer_name in ("best.json", "latest.json"):
        pointer_path = checkpoint_path / pointer_name
        if not pointer_path.is_file():
            continue
        return _resolve_pointer(pointer_path)
    return str(checkpoint_path)


def _resolve_pointer(pointer_path: Path) -> str:
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    target = pointer.get("checkpoint") or pointer.get("path") or pointer.get("directory")
    if not isinstance(target, str) or not target:
        raise ValueError(f"{pointer_path} must contain a checkpoint path")
    resolved = Path(target)
    if not resolved.is_absolute():
        resolved = pointer_path.parent / resolved
    if not resolved.is_dir():
        raise FileNotFoundError(f"checkpoint pointer {pointer_path} refers to missing directory {resolved}")
    return str(resolved)


def _installed_kev_revision() -> str | None:
    distribution = importlib.metadata.distribution("kev")
    direct_url_text = distribution.read_text("direct_url.json")
    if not direct_url_text:
        return None
    direct_url = json.loads(direct_url_text)
    return direct_url.get("vcs_info", {}).get("commit_id")


def _assert_trainable_parameters(model: object) -> None:
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not any(name.startswith("head.") for name, _ in trainable):
        raise RuntimeError("Kev pointer head has no trainable parameters")
    if not any("lora_" in name or "trainable_tokens_delta" in name for name, _ in trainable):
        raise RuntimeError("Kev checkpoint adapter has no trainable parameters")
    unsupported = [
        name
        for name, _ in trainable
        if not name.startswith("head.") and "lora_" not in name and "trainable_tokens_delta" not in name
    ]
    if unsupported:
        raise RuntimeError(f"unexpected trainable base-model parameters: {unsupported[:3]}")
