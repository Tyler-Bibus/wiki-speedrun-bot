# wiki-speedrun-bot

An untuned Kev-0.8B baseline for Wikispeedia: give the model a target, the current
article, and its outgoing links, then let it choose the next click. The first goal
is to measure whether the released model can navigate before training anything.

The Python application uses the standard library. Kev runs in its own local GPU
server, using the upstream model definitions and loader. There are no custom
model weights, supervised training loops, or RL updates yet.

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

## Later training

If the baseline is weak, use supervised fine-tuning / imitation learning from
human navigation examples, starting from released Kev weights. Split complete
trajectories and start/target pairs before turning them into individual decisions.
Keep development and held-out test tasks fixed for comparisons.

After that, add `training/sft.py`, `training/rollouts.py`, `training/rewards.py`,
`training/rl.py`, and `models/trainable_kev.py`. RL needs in-process access to
model gradients; the HTTP inference client cannot perform policy updates. A larger
Kev checkpoint can be compared through the same environment and evaluation loop.
