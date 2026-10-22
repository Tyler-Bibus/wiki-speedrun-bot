#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export HF_HOME="$repo_root/.cache/huggingface"
export UV_CACHE_DIR="$repo_root/.cache/uv"
export TRITON_CACHE_DIR="$repo_root/.cache/triton"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$TRITON_CACHE_DIR"

if [[ -x .venv/bin/python ]]; then
    existing_python="$(.venv/bin/python --version 2>&1)"
    if [[ "$existing_python" != "Python 3.12."* ]]; then
        echo "Existing .venv uses $existing_python; remove it before creating the Python 3.12 training environment." >&2
        exit 1
    fi
fi

uv python install 3.12
uv sync --locked --python 3.12 --extra train
.venv/bin/python scripts/check_training.py
