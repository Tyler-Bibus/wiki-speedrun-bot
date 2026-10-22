# Training Kev navigation in WSL

The training pipeline continues the released Kev-0.8B LoRA adapter and pointer
head with supervised examples, then on-policy REINFORCE. The base weights stay
frozen. The [implementation plan](rl-plan.md) explains the algorithm and file
formats in detail.

Defaults are a bounded pilot for a personal computer: sample up to 2,000 human
paths, run SFT for at most 200 optimizer updates or one epoch, then run 20 RL
updates with four episodes each. The commands below use these limits automatically.

Run commands from the repository root. The Linux setup selects Python 3.12 and
installs the optional GPU dependencies into `.venv`; it does not use Windows
virtual environments or the system Python 3.14.

```bash
uv python install 3.12
bash scripts/setup_training.sh
.venv/bin/python scripts/check_gpu.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/smoke_training.py
```

The setup pins the Kev source revision and CUDA PyTorch in `uv.lock`. Caches
live in `.cache/`. The first model load downloads the base weights and released
adapter. Training happens in process; the HTTP inference server is not needed.

## Prepare the dataset

```bash
.venv/bin/python -m wiki_speedrun prepare-training --config configs/training.toml
```

This reads the copied SNAP graph and completed human paths from `data/`. If
needed, download them with the existing `prepare-data` command first. Data
preparation removes backtracking to create forward routes, rejects routes with
illegal edges, and assigns whole start/target pairs to frozen train/dev/test
splits before creating individual supervised decisions. The ten baseline pairs
are reserved for development.

`data.max_paths = 2000` selects a reproducible sample before expanding routes
into decisions. `prepare-training --max-paths N` explicitly overrides that sample
size. The complete raw graph and article text remain available for navigation.

Prepared artifacts are UTF-8 JSON/JSONL with a versioned manifest containing
hashes and rejection counts. Keep them with the checkpoints for reproducible
comparisons. Use `--force` only when intentionally rebuilding prepared files.
For a separate smoke dataset, use separate output paths in a separate config.
Changing the sample size requires intentionally rebuilding the prepared files.

## Run SFT and RL

```bash
.venv/bin/python -m wiki_speedrun train-sft --config configs/training.toml
.venv/bin/python -m wiki_speedrun train-rl --config configs/training.toml --init-from checkpoints/SFT_RUN_ID/best.json
```

Replace the example run ID with the checkpoint location printed by SFT.
`configs/training.toml` controls the epoch/update budgets, token limits, task
batch size, reward settings, and checkpoint/evaluation intervals. For a short
SFT check, add `--max-updates 2 --max-examples 16 --max-tasks 2`. For a short RL check, add
`--max-updates 2 --max-tasks 2`. Small limits validate mechanics rather than
navigation quality. The update limit is a total optimizer-update ceiling,
including completed updates when resuming.

`sft.max_updates = 200` applies when the SFT command has no `--max-updates` flag;
the flag explicitly overrides it. Increase the budget deliberately after judging
the pilot. RL defaults to 20 updates with two task pairs and two episodes per pair.

For an overnight SFT run, use the larger profile explicitly:

`configs/overnight.toml` allows two epochs, capped at 2,000 optimizer updates.
It keeps the same 2,000-path sample, batch sizes, and checkpoint retention as
the pilot. It runs SFT only; start RL separately after reviewing development
results. The local RTX 3090 check measured about seven seconds per update;
budget roughly 4–6 hours including periodic evaluation, depending on GPU load.

```bash
.venv/bin/python -m wiki_speedrun train-sft --config configs/overnight.toml
```

Before committing to the full run, a quick check can use `--max-updates 10` with
the same profile, then resume from the printed checkpoint:

```bash
.venv/bin/python -m wiki_speedrun train-sft --config configs/overnight.toml --max-updates 10
.venv/bin/python -m wiki_speedrun train-sft --config configs/overnight.toml --resume checkpoints/SFT_RUN_ID/latest.json
```

Keep the profile identical when resuming; adding `--max-examples` or `--max-tasks`
changes the selection fingerprint. Checkpoint retention is limited per run. On
Windows/WSL, keep the host awake and leave the terminal open while training runs.

Training evaluates a fixed seeded sample of at most 16 development tasks every
100 updates and at the final update (SFT also evaluates at epoch boundaries).
Set `save.evaluation_max_tasks` and `save.evaluate_every_updates` to adjust this
budget. Standalone `evaluate-trained` uses the full requested split unless you
pass `--max-tasks`. This keeps routine development checks from dominating the
training time as the dataset grows.

Default reward for one navigation click is `-0.02`, with an additional `+1`
on success or `-1` on a terminal failure. A two-click success scores `0.96`;
a ten-click success scores `0.80`. The twelve-click cap and validated reward
configuration ensure any permitted success beats a failure.

Kev handles large outgoing-link sets through a tournament: choose a winner
from each token-safe group, then choose among winners until one link remains.
No outgoing link is dropped during preprocessing. A 294-link page typically
uses five first-round predictions and one final prediction with 64-choice
groups. That is one navigation click and six model calls. Training records
every sampled internal choice and applies its click's return to all those
decisions, including earlier winners that later lose.

RL collects fresh sampled episodes under a fixed policy, recomputes their
log probabilities with gradients, and takes one optimizer step. It then
discards the batch. Dropout is off during collection and replay. This keeps
the sampled policy and gradient calculation consistent.

## Checkpoints, recovery, and evaluation

Each checkpoint includes an upstream-compatible adapter in Safetensors,
`head.pt`, tokenizer files, and local `training_state.pt` with optimizer,
cursor, and RNG state. JSON manifests identify the data/configuration and
completed logs. `latest.json` and `best.json` point to completed checkpoint
directories. Files are saved atomically at optimizer boundaries.

`save.checkpoint_every_updates = 50` controls routine saves; evaluation improvements
and the final update can also save. After a successful save and pointer update,
`save.keep_last_checkpoints = 2` retains the two newest committed checkpoints
plus any older checkpoint referenced by `best.json`. This normally leaves at
most three checkpoints per run, approximately **450 MiB** at the measured
149 MiB/checkpoint size. Atomic saving temporarily needs room for another copy.
Both SFT and RL use this policy.

Retention applies within the current run; separate runs, datasets, logs, and
the shared model/CUDA dependency cache use additional space. Cleanup skips
unfinished, foreign, or symlinked directories and preserves pointer targets.

```bash
.venv/bin/python -m wiki_speedrun train-rl --config configs/training.toml --resume checkpoints/RL_RUN_ID/latest.json
.venv/bin/python -m wiki_speedrun evaluate-trained --config configs/training.toml --checkpoint checkpoints/RL_RUN_ID/best.json --split dev
.venv/bin/python -m wiki_speedrun evaluate-trained --config configs/training.toml --checkpoint jaredpalmer/kev-0.8b@v1.0 --split baseline
```

Resume restores weights, optimizer, sampling state, and the committed log
boundary, and rejects incompatible data/configuration. Use `--init-from` to
start a new run from weights instead of resuming its optimizer and cursor.
The smaller defaults change configuration fingerprints: checkpoints made with
older settings require their original config for an exact resume.

Compare trained and released checkpoints using identical tournament and
observation settings; retain the old HTTP baseline as a separate historical
comparison. Report success first, then clicks on successful episodes, shortest
path overhead, failure reasons, model-call count, and latency. Evaluate on test
pairs only after selecting the training approach and checkpoint on development
pairs. A smoke run or the ten-task baseline alone does not establish a gain.

Run logs under `runs/` use JSONL for completed transitions, episodes, and
updates, plus resolved configuration, provenance, aggregate metrics, and an
error artifact on infrastructure failure. Infrastructure errors abort training;
they are never converted into negative navigation rewards. Data, caches, runs,
and checkpoints remain ignored by Git.
