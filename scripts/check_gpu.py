"""Print and exercise PyTorch CUDA details for the current Python environment."""

import sys

try:
    import torch
except ImportError:
    print(f"Python: {sys.executable}")
    print("PyTorch is not installed in this Python environment.")
    print("Run scripts/setup_kev.ps1, then rerun this check with the prepared Python.")
    raise SystemExit(1)

print(f"Python: {sys.executable}")
print(f"PyTorch: {torch.__version__}")
if not torch.cuda.is_available():
    print("CUDA available: False")
    print("No CUDA GPU is visible to this PyTorch build.")
    raise SystemExit(1)

print(f"CUDA available: True (runtime {torch.version.cuda})")
try:
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        memory_gib = props.total_memory / (1024 ** 3)
        try:
            bf16 = torch.cuda.is_bf16_supported()
        except (AttributeError, RuntimeError):
            bf16 = "unknown"
        print(f"GPU {index}: {props.name}")
        print(f"  Memory: {memory_gib:.1f} GiB")
        print(f"  bfloat16 supported: {bf16}")

    # Force context initialization, an actual kernel launch, and synchronization.
    sample = torch.ones((64, 64), device="cuda")
    result = sample @ sample
    torch.cuda.synchronize()
    print(f"CUDA matrix multiply: OK ({result[0, 0].item():.0f})")
except Exception as exc:
    print(f"CUDA runtime check failed: {type(exc).__name__}: {exc}")
    raise SystemExit(1)
