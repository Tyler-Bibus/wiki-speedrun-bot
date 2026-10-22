# wiki-speedrun-bot

Kev-0.8B navigation for Wikispeedia: give the model a target, the current article,
and its outgoing links, then let it choose the next click. The project includes
the original untuned baseline and optional supervised fine-tuning and RL loops.

The baseline application uses the standard library and calls a local Kev GPU
server. Training uses the optional PyTorch environment and loads Kev in process,
continuing the released adapter and pointer head while freezing the base model.
See [training in WSL](docs/training.md) for setup, commands, and saved formats.

## Quick start on Windows with an NVIDIA GPU

Use Python 3.12 or 3.13, Git, and a working NVIDIA driver. Run these commands from
the repository root in PowerShell. Setup downloads the upstream source and
dependencies, including a roughly 3 GB CUDA PyTorch wheel. The first server launch
also downloads the base model and Kev adapter/head. All caches stay in `.cache/`.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\setup_kev.ps1
.venv\Scripts\python.exe -m wiki_speedrun prepare-data
```

In a second terminal, start the model server and wait for its startup message:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\serve_kev.ps1
```

Back in the first terminal:

```powershell
.venv\Scripts\python.exe -m wiki_speedrun smoke-test
.venv\Scripts\python.exe -m wiki_speedrun play Earth Moon
.venv\Scripts\python.exe -m wiki_speedrun make-tasks --count 10
.venv\Scripts\python.exe -m wiki_speedrun evaluate
```

`Earth -> Moon` is an intentionally easy one-click sanity check. The generated
tasks have shortest paths of two or three clicks. These tasks are development
examples, not a held-out test set. A successful run on a few examples is only a
proof of concept.

Stop the server with Ctrl+C when finished. It listens on `127.0.0.1:8008`.

### Run without a model

To inspect the navigation pipeline without installing GPU dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -e .
.venv\Scripts\python.exe -m wiki_speedrun prepare-data
.venv\Scripts\python.exe -m wiki_speedrun play Earth Moon --policy random
```

The random policy samples uniformly from outgoing links. Its results are marked
`random`; it does not use Kev or indicate model performance.

## Settings and saved results

Edit `configs/baseline.toml` to change the endpoint, checkpoint, random seed,
article excerpt size, or click limit (12 by default). Paths are relative to the
working directory. You can select another config:

```powershell
.venv\Scripts\python.exe -m wiki_speedrun --config configs/baseline.toml play Apple Fruit
```

The server launcher reads the checkpoint from `configs/baseline.toml`. When using
a different inference config or checkpoint, pass `-Checkpoint` and `-Port` to
`serve_kev.ps1` as needed, and set the matching client endpoint in your config.

Each completed command writes a timestamped folder under `runs/` containing:

- `episodes.jsonl`: full navigation paths, selected links, available choices,
  scores, confidence values, request latency, and termination reasons.
- `metrics.json`: success rate, clicks, revisits, inference latency, and excess
  clicks over a shortest path for successful episodes.
- `config.json`: the settings and policy used for the run.

An episode ends on success, a dead end, the click limit, or an explicit
`option_limit` result. There is no automatic target-link selection, graph search
inside the policy, or filtering of visited pages. Shortest-path search is used
only for selecting tasks and reporting metrics. The model sees the current page
excerpt, current/target titles, history, and outgoing link titles.

Kev accepts at most 255 choices per question. A page above the configured limit
ends that episode with `option_limit`; its partial trace is preserved. Generated
tasks have a route within the limit, but the model can still wander to a page
that exceeds it. Long inputs can also exceed the upstream server's token budget;
those errors are reported instead of silently truncating choices.

## Data

The [Stanford SNAP Wikispeedia dataset](https://snap.stanford.edu/data/wikispeedia.html)
contains a fixed graph of 4,604 articles and 119,882 directed links, human paths,
and article text. This prototype navigates that local graph rather than today's
live Wikipedia. `prepare-data` downloads the graph/path archive and plaintext
archive, then writes `data/processed/graph.json`.

Use `prepare-data --no-text` for a graph-only run. Article IDs retain underscores;
commands also accept quoted titles with spaces. Human path parsing is included
for exploration: `load_paths` returns each path's final active stack after
backtracking, not the full click history. Do not use that representation directly
as chronological imitation-learning examples.

## Files

```text
configs/baseline.toml       Experiment settings
scripts/setup_kev.ps1       Install pinned upstream Kev and CUDA PyTorch
scripts/serve_kev.ps1       Start the prepared model server
scripts/check_gpu.py       Verify PyTorch sees the GPU and can run a CUDA operation
src/wiki_speedrun/
  cli.py                   prepare-data, smoke-test, make-tasks, play, evaluate
  config.py                Load TOML settings
  data/download.py         Download and safely extract SNAP archives
  data/load.py             Load articles, directed links, text, and human paths
  data/tasks.py            Select reachable pairs and compute shortest paths
  environment.py           Navigation state and legal moves
  observations.py          Build model requests
  models/kev.py            Local HTTP client and model smoke test
  policy.py                Kev and random action selection
  runner.py                Run an episode and record decisions
  evaluation.py            Aggregate outcomes
tests/                     Data, environment, HTTP, and CLI tests
```

Downloaded data, environments, model weights, and run outputs are ignored by Git.
The application package has no runtime dependencies; `uv.lock` locks that small
package. Kev's separate environment uses the upstream release lock, then replaces
its Windows CPU-only PyTorch with `torch==2.8.0+cu128`. Start it with the provided
launcher rather than `uv run`, which could restore the CPU-only wheel.

The setup pins upstream Kev source revision
`6b719c3c3f367295f6ef336f4f751cf5ff970abc` (`kev-1.0`) and checkpoint
`jaredpalmer/kev-0.8b@v1.0`. Fused kernels and CUDA graphs are disabled by default
in the launcher to keep the initial Windows execution path simple. The model
still runs on the GPU. See the [upstream Kev instructions](https://github.com/jaredpalmer/kev)
and [model card](https://huggingface.co/jaredpalmer/kev-0.8b).

## Tests

```powershell
.venv\Scripts\python.exe -m unittest discover -s tests -v
.cache\kev\.venv\Scripts\python.exe scripts/check_gpu.py
```

The unit tests use small fixtures and a local fake HTTP endpoint. The separate
`smoke-test` and `play` commands exercise the actual loaded model.

The [first prototype results](docs/baseline-results.md) record an actual RTX 3090
run: the one-click check succeeded, and untuned Kev solved 1/10 sampled longer
tasks versus 0/10 for random navigation. The fixed pairs are saved under
`examples/baseline_tasks.json` for repeatable comparisons.

## Training in WSL

Use Python 3.12 and the pinned Linux training environment:

```bash
bash scripts/setup_training.sh
.venv/bin/python -m wiki_speedrun prepare-training --config configs/training.toml
.venv/bin/python -m wiki_speedrun train-sft --config configs/training.toml
.venv/bin/python -m wiki_speedrun train-rl --config configs/training.toml --init-from checkpoints/SFT_RUN_ID/best.json
```

Replace `SFT_RUN_ID` with the SFT checkpoint location printed by the command.
Training dependencies and model weights are prepared in `.venv` and `.cache`.
The existing Windows scripts continue to support the original inference baseline.

The [training guide](docs/training.md) covers short validation runs, resume,
evaluation, and configuration. The reward is `+1` for success, `-1` for terminal
failure, and `-0.02` per navigation click. Large choice sets use token-safe
tournaments: predictions select group winners, then another prediction selects
among the winners. Every outgoing link enters the tournament.

The [implementation plan](docs/rl-plan.md) documents the interfaces, file formats,
and Luna agent assignments. The [initial readiness review](docs/training-readiness.md)
records the pre-implementation findings. The [training validation report](docs/training-validation.md)
records the tests and short RTX 3090 checks; those checks establish working
training mechanics, rather than a measured navigation improvement.
The [supervised model results](docs/sft-results.md) record the completed
overnight SFT run and full validation/test navigation evaluation of its saved
best checkpoint.

For RL from that selected SFT model, use `configs/rl.toml`. The
[RL smoke report](docs/rl-smoke-results.md) records the GPU update/reload/resume
checks and runtime estimate; the [training guide](docs/training.md#rl-after-the-completed-sft-run)
has launch, resume, and comparison commands. RL uses a new checkpoint run and
preserves the untuned comparison copy and all saved checkpoints from the completed SFT run.
The [optimized-kernel report](docs/training-kernels.md) records the faster validated
training stack and updated runtime estimate.


# Future Val
Later on
In addition, I would like to use this evaluation framework to try to use the actual Jev model as an option using my API key for openRouter [https://openrouter.ai/typesafe/jev-1.14?view=api](https://openrouter.ai/typesafe/jev-1.14?view=api) (Create a location for me to paste my key that is git ignored). I want to compare my fine tuned against it.

Also I should consider comparing 0.8 fine tuned against other sized decision and maybe even LLMs using openrouter.

I should also compare different fine tuning/LorA/Transfer Learning methods.
