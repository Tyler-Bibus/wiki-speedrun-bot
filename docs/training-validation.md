# Training implementation validation

The subsequent overnight SFT run completed 2,000 updates. Its selected
checkpoint and full validation/test evaluation are documented in the
[supervised model results](sft-results.md). The checks below describe the earlier
implementation and readiness validation.

Validated October 21, 2026 in WSL on the RTX 3090 (24 GiB). Implementation used
Luna agents for the model/environment, choices/data, and learning loops, followed
by coordinating-agent review and integration.

The subsequent personal-computer settings add checkpoint retention and smaller
preparation/training budgets; all **52 tests** pass after these changes. Retention
tests protect best/latest and resume, and cover failed saves, invalid pointers,
and unfinished directories. Tests also verify configured SFT/sample limits and
explicit CLI overrides. Current defaults are in the
[training guide](training.md); the smoke results below describe the original
mechanics checks.

The Linux environment uses Python 3.12.15, CUDA PyTorch 2.8.0+cu128, Transformers
5.20.2, PEFT 0.21.2, and Kev source revision
`6b719c3c3f367295f6ef336f4f751cf5ff970abc`. The optional dependency set is locked
in `uv.lock`. `scripts/check_training.py` verified those core pins and completed
a CUDA matrix multiplication.

## Checks completed

- All 48 baseline/training tests passed, including pair splits,
  strict input admission, 294-choice tournaments, reward boundaries, SFT/RL
  updates, checkpoint hashes, and preservation of earlier evaluation traces
  after a later task fails.
- Two-update CPU tests compare uninterrupted SFT/RL against stop-and-resume:
  final weights match exactly. RL also reproduces sampled task pairs and
  tournament choices. A no-op SFT resume leaves logs unchanged.
- `uv lock --check --offline --python 3.12`, Python compilation, shell syntax,
  and whitespace checks passed.
- `scripts/smoke_training.py` loaded released Kev on CUDA, compared raw logits
  to upstream's unmerged loader, and performed three optimizer steps on a tiny
  fixed example. Adapter and pointer-head gradients were finite and nonzero;
  frozen base parameters had no gradients, and the checked base weights stayed
  unchanged. The tiny example's loss decreased. Exported weights reloaded with
  matching logits (`atol=1e-3`, `rtol=1e-4`). Peak allocated CUDA memory was
  approximately 5.74 GiB during this check, not an estimate for full training.
- The same real-model check processed `United_States` with all 294 choices,
  produced one legal final link through six predictions, and retained the
  internal decision trace.
- CLI data preparation used an isolated sample of 64 source paths. Under a
  four-click smoke limit, 32 routes were retained, producing 107 supervised
  training nodes, 17 development nodes, and 24/14/3 train/dev/test task pairs.
  The other 32 paths exceeded that smoke limit. Whole-graph token preflight
  checked 4,565 pages with at least two choices; its worst conservative pair
  bound was 1,099 tokens within the 4,096-token row budget.
- Real CLI SFT completed optimizer updates, wrote upstream-compatible weights
  and best/latest pointers, and resumed from a saved optimizer boundary. RL
  initialized from SFT, completed an update with a finite nonzero gradient,
  then resumed for a second update without log-probability replay mismatches.
- Standalone checkpoint evaluation writes incremental transitions with full
  tournament requests, episodes, resolved configuration, provenance, and metrics.

## Local artifacts

The ignored `.cache/training-smoke/` directory contains the smoke configuration,
prepared dataset/manifest, wrapper check report, and timestamped SFT/RL/evaluation
runs. Smoke checkpoints are under `checkpoints/` and identify their small dataset
and selection limits through manifests/provenance. They are mechanical checks,
not trained checkpoints selected for use on the full task distribution.

- [Real wrapper/GPU check](../.cache/training-smoke/wrapper-check.json)
- [Final CLI save/resume checks](../.cache/training-smoke/final-cli-checks.json)
- [SFT update metrics](../.cache/training-smoke/runs/20261022T022653.772057Z-sft/metrics.jsonl)
- [RL update metrics](../.cache/training-smoke/runs/20261022T022727.033400Z-rl/metrics.jsonl)

The original smoke run did not populate the default dataset paths. The overnight
readiness check below subsequently prepared them using the bounded 2,000-path
sample. The [training guide](training.md) gives commands and explains budgets.

The correctness checks do not establish improved navigation quality. At that stage, no long
training run or held-out test evaluation had been performed. The short RL run used
four-click episodes and a tiny training pool; compare properly trained weights
to released Kev under identical preprocessing on frozen development/test pairs.

Training deliberately uses upstream's reference PyTorch kernels. Optional fused
serving kernels and CUDA graphs are not required. Throughput and peak memory on
larger batches or contexts should be measured before increasing those limits.

## Overnight readiness check

The same RTX 3090 passed `scripts/check_training.py` and a real SFT run with
`configs/overnight.toml`: ten optimizer updates, development evaluation, then a
checkpoint resume through update eleven and another evaluation. Losses and
gradients were finite, and the update log contained exactly updates 1–11.
The overnight run had not yet started at this readiness check; it has since
completed. See the [supervised model results](sft-results.md).

The prepared 2,000-path sample retained 1,968 valid routes and produced 10,001
training and 1,180 development decision records. Task splits contain 1,404
training, 179 development, and 175 test pairs. The prepared decisions occupy
about 121 MiB; the graph token preflight passed.

The first ten updates averaged 7.07 seconds each, excluding loading and
evaluation. Peak PyTorch allocated GPU memory was 10.17 GiB (10.55 GiB reserved).
Each 16-task evaluation took about 42 seconds. A continuation to 2,000 updates
should take roughly 4–6 hours at this speed. Other GPU use and longer examples
can change timing and memory. Checkpoints measured 149.34 MiB each; retention
normally limits this run to about 450 MiB plus temporary save space. Shared
model/dependency caches, other runs, and logs use additional disk space.

Continue the tested run from the repository root:

```bash
.venv/bin/python -m wiki_speedrun train-sft \
  --config configs/overnight.toml \
  --resume checkpoints/20261022T025733.517853Z-sft/latest.json
```

Keep Windows awake and the terminal open. Use the same configuration and
prepared dataset for resume. The two short development evaluations each solved
13 of 16 tasks; this is a mechanics check, not evidence of improvement without
a matching released-model comparison on the same tasks and settings.

- [Readiness measurements](../.cache/overnight-readiness.json)
- [Update and evaluation metrics](../runs/20261022T025733.517853Z-sft/metrics.jsonl)
