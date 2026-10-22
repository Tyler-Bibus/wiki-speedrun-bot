"""Check the pinned Kev source and execute a real CUDA operation."""

from __future__ import annotations

import importlib.metadata
import json
import sys


EXPECTED_TORCH = "2.8.0+cu128"
EXPECTED_KEV_REVISION = "6b719c3c3f367295f6ef336f4f751cf5ff970abc"


def main() -> int:
    if sys.version_info[:2] != (3, 12):
        print(f"Expected Python 3.12, got {sys.version.split()[0]}", file=sys.stderr)
        return 1

    try:
        import torch
    except ImportError as exc:
        print(f"PyTorch is missing from {sys.executable}: {exc}", file=sys.stderr)
        return 1
    if torch.__version__ != EXPECTED_TORCH:
        print(f"Expected torch {EXPECTED_TORCH}, got {torch.__version__}", file=sys.stderr)
        return 1

    distribution = importlib.metadata.distribution("kev")
    direct_url_text = distribution.read_text("direct_url.json")
    direct_url = json.loads(direct_url_text) if direct_url_text else {}
    installed_revision = direct_url.get("vcs_info", {}).get("commit_id")
    if installed_revision != EXPECTED_KEV_REVISION:
        print(
            f"Expected Kev revision {EXPECTED_KEV_REVISION}, got {installed_revision!r}",
            file=sys.stderr,
        )
        return 1

    print(f"Python: {sys.executable} ({sys.version.split()[0]})")
    print(f"PyTorch: {torch.__version__}; CUDA build {torch.version.cuda}")
    print(f"Kev source revision: {installed_revision}")
    if not torch.cuda.is_available():
        print("CUDA is unavailable to this PyTorch environment.", file=sys.stderr)
        return 1

    try:
        sample = torch.ones((64, 64), device="cuda")
        product = sample @ sample
        torch.cuda.synchronize()
    except Exception as exc:
        print(f"CUDA operation failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    properties = torch.cuda.get_device_properties(0)
    print(f"GPU: {properties.name}; memory {properties.total_memory / (1024 ** 3):.1f} GiB")
    print(f"CUDA matrix multiply: OK ({product[0, 0].item():.0f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
