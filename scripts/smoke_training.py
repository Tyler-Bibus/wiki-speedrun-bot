"""Exercise real Kev gradients and checkpoint reload on one tiny fixed example."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import tempfile


def main() -> None:
    os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache/huggingface"))
    import torch
    from kev.checkpoint import Checkpoint, LoadOptions

    from wiki_speedrun.models.trainable_kev import TrainableKev
    from wiki_speedrun.training.config import ChoiceConfig
    from wiki_speedrun.choices import TournamentPolicy
    from wiki_speedrun.data.load import load_graph
    from wiki_speedrun.environment import NavigationEnvironment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="jaredpalmer/kev-0.8b@v1.0")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("The real-model smoke check requires CUDA")
    torch.manual_seed(42)
    torch.cuda.reset_peak_memory_stats()
    model = TrainableKev.from_checkpoint(args.checkpoint, "cuda", ChoiceConfig())
    model.eval()
    request = {
        "state": {"current_title": "Earth", "target_title": "Moon",
                  "current_excerpt": "Earth has one natural satellite, the Moon.",
                  "history": ["Earth"]},
        "questions": {"next_link": {
            "type": "choice",
            "instructions": "Choose the outgoing link that advances toward the target.",
            "criteria": {"link_0": "Moon", "link_1": "Sun", "link_2": "Mars"},
        }},
    }
    encoded = model.encode(request)
    with torch.no_grad():
        initial_logits = model.logits(encoded).clone()

    # Compare the differentiable path against upstream's unmerged raw-logit loader.
    options = LoadOptions(merge=False, temperature=1.0, backend="torch",
                          dtype=torch.float32, fused=False, cuda_graphs=False)
    _, reference = Checkpoint(args.checkpoint).load("cuda", options)
    with torch.no_grad():
        reference_logits = reference.forward(encoded["encoded"])[0].float()
    torch.testing.assert_close(initial_logits, reference_logits, atol=1e-3, rtol=1e-4)
    del reference, reference_logits
    gc.collect()
    torch.cuda.empty_cache()

    named_parameters = list(model.model.named_parameters())
    trainable = [(name, parameter) for name, parameter in named_parameters if parameter.requires_grad]
    frozen = next(parameter for _, parameter in named_parameters if not parameter.requires_grad)
    frozen_before = frozen.detach().reshape(-1)[:64].clone()
    before = {name: parameter.detach().clone() for name, parameter in trainable}
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=2e-4)
    target = torch.tensor([0], device="cuda")
    losses = []
    gradient_groups = set()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model.logits(encoded)
            loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), target)
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite smoke loss")
        losses.append(float(loss.detach()))
        loss.backward()
        for name, parameter in trainable:
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"Non-finite gradient: {name}")
                if torch.count_nonzero(parameter.grad):
                    gradient_groups.add("head" if name.startswith("head.") else "adapter")
        torch.nn.utils.clip_grad_norm_([parameter for _, parameter in trainable], 1.0)
        optimizer.step()
    assert gradient_groups == {"adapter", "head"}, gradient_groups
    assert all(parameter.grad is None for _, parameter in named_parameters if not parameter.requires_grad)
    assert torch.equal(frozen_before, frozen.detach().reshape(-1)[:64])
    assert any(not torch.equal(before[name], parameter.detach()) for name, parameter in trainable)
    assert losses[-1] < losses[0], losses
    with torch.no_grad():
        final_logits = model.logits(encoded).clone()
    output_root = Path(".cache/training-smoke")
    output_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="weights-", dir=output_root) as temporary:
        model.save_checkpoint(Path(temporary))
        del optimizer, trainable, named_parameters, before, frozen, model, logits, loss
        gc.collect()
        torch.cuda.empty_cache()
        restored = TrainableKev.from_checkpoint(temporary, "cuda", ChoiceConfig()).eval()
        with torch.no_grad():
            reloaded_logits = restored.logits(restored.encode(request))
        torch.testing.assert_close(final_logits, reloaded_logits, atol=1e-3, rtol=1e-4)
        graph = load_graph(args.data_dir)
        env = NavigationEnvironment(graph, "United_States", "Moon", max_steps=1)
        decision = TournamentPolicy(restored, ChoiceConfig()).choose(env)
        assert decision["article"] in env.legal_moves()
        assert len(decision["nodes"]) == 6
    report = {
        "checkpoint": args.checkpoint,
        "steps": args.steps,
        "losses": losses,
        "adapter_and_head_gradients": True,
        "frozen_base_unchanged": True,
        "upstream_logits_match": True,
        "checkpoint_reload_matches": True,
        "large_page_choices": len(env.legal_moves()),
        "large_page_model_calls": len(decision["nodes"]),
        "peak_cuda_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
    }
    (output_root / "wrapper-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
