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

## OpenRouter comparison

The remote benchmark uses three paid OpenRouter models: Jev 1.14 for structured
decisions, DeepSeek V4.1 Flash 1017, and MiniMax M3.1 Flash.
It uses the existing environment and frozen Wikispeedia splits. No installation,
training, or environment upgrade is performed by the benchmark command.

Set `OPENROUTER_API_KEY` in your process environment or in the ignored
`.env.openrouter` file. `.env.openrouter.example` shows the format. The file
accepts comments, blank lines, and exactly one plain or quoted assignment;
it is parsed as data. A nonempty environment key takes precedence. Other
commands do not read these credentials.

From the repository root, reproduce all requested remote systems and the three
pinned local checkpoints, validate the saved evidence, and generate reports:

```bash
.venv/bin/python -m wiki_speedrun benchmark
```

Results, progress, and reports are in `runs/openrouter-paid-comparison/`. Run the
same command again to resume. Completed episodes, including invalid LLM
responses, are preserved. An interrupted episode restarts; its earlier traces
and API charges remain recorded. Resume checks code, config, model snapshot,
tokenizer, graph/text, and ordered tasks. Budget limits may be raised explicitly
on an individual `evaluate --resume` command; protocol settings cannot change.
Use a new `--output` directory when changing suite settings or source code.

For a bounded five-pair development reproduction and its report:

```bash
.venv/bin/python -m wiki_speedrun benchmark --max-tasks 5 --splits dev --output runs/openrouter-five-task
```

The default lineup is `minimax deepseek jev`. Use `--models jev` to run only
the decision model. `--no-include-local` avoids
local GPU evaluation. Existing local artifacts are reused only after checking
checkpoint hashes, graph/text, frozen task coverage, transitions, and every
recorded tournament group and token count. Missing local runs are evaluated
without optimizer updates. The local checkpoints are fixed in
`src/wiki_speedrun/benchmark.py`; no moving `best.json` pointer is used.

Decision systems use `matched_kev_tournament`, with the exact pinned Kev CPU
encoder, preserved local tokenizer, 64-choice node cap, 1,024 state tokens,
4,096 row tokens, 1,500-character excerpts, and 12 clicks. The encoder needs
only the existing optional `tokenizers` and Kev packages; it loads no model
weights, PyTorch, or CUDA. LLMs send the full ordered link list and complete
history in one request, without any local tokenizer or tournament.
DeepSeek uses `llm_full_list_json` with strict JSON schema. MiniMax currently
lacks strict-schema support and is explicitly configured as
`llm_full_list_json_object`: JSON-object mode with the same strict local
validation. This is a declared change from the original plan. No automatic
format downgrade or model fallback occurs. Invalid JSON or an illegal article
ends the episode immediately and remains in the denominator.

Each remote config bounds requests and reported cost; requests run serially.
All three models use paid endpoints, so the shared free-model daily quota does
not apply. Provider rate limits may still stop a run. At most two retries are
allowed for 429 and selected 5xx responses, with at most 30 seconds of backoff
per retry. Explicit provider 429/5xx errors
inside HTTP-success responses use the same bounded retries; repeated provider
failures also trigger the background worker's five-minute resume cycle. Choice
distributions allow the API's two-decimal rounding across all returned options
while retaining the original probabilities and selected choice. Ambiguous
network timeouts are never retried automatically. The final request may overshoot the cost limit;
missing charges are unknown, and request limits remain in effect. A full run
can take hours. Saved costs include failed and abandoned attempts;
probe costs are reported separately. Network latency and local GPU latency
have different execution conditions.

Individual commands retain global `--config` placement:

```bash
.venv/bin/python -m wiki_speedrun --config configs/jev.toml smoke-test --policy decision
.venv/bin/python -m wiki_speedrun --config configs/jev.toml play Earth Moon --policy decision
.venv/bin/python -m wiki_speedrun --config configs/llm.toml evaluate --policy llm --split dev --max-tasks 5
.venv/bin/python -m wiki_speedrun --config configs/minimax.toml evaluate --policy llm --split test
.venv/bin/python -m wiki_speedrun --config configs/jev.toml evaluate --policy jev --split dev
```

`--tasks` and `--split` are mutually exclusive. Remote evaluation defaults to
the development split; `--max-tasks` always labels a subset. `--credentials-file`,
`--model`, `--max-requests`, `--max-cost`, and `--resume RUN_DIRECTORY` are
available for remote evaluation. A complete run has config/provenance,
per-attempt requests, legal transitions, completed episodes, rejected LLM
decisions, status, and metrics. Incomplete runs never enter paired comparisons.
Standalone comparison is available through `compare-runs RUN_DIRECTORY ...
--output runs/report.md`; different protocols require
`--allow-system-protocols` explicitly.

Run the offline regression suite in one line:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Time a plain BFS baseline on the same frozen evaluation pairs, offline:

```bash
.venv/bin/python scripts/benchmark_bfs.py --training-config configs/snap-full.toml --splits dev test --output runs/bfs-snap-full.json
```

For the original splits, replace `--training-config configs/snap-full.toml`
with `--config configs/jev.toml`. Match the model evaluation's `--splits` and
`--max-tasks` when comparing subsets; `--tasks FILE` accepts explicit custom
pairs. JSON includes total `wall_seconds`, mean time per task, median/p95 query
time, per-pair paths and timing, success within the configured click limit,
and graph/task hashes. Existing output files are never overwritten.

Each task runs a fresh directed BFS with no cached search results. Timing
includes path reconstruction and per-task bookkeeping; loading, manifest
validation, hashing and JSON output are excluded. Graph loading is timed
separately. Compare total wall time with a model run's `wall_seconds` only for
the same graph and complete task coverage. Summed API/inference latency is a
different measure, particularly with concurrent remote episodes. BFS sees the
entire graph, so this is an algorithm baseline with full graph access. The
script makes no API calls and leaves evaluator source, settings, and saved
runs unchanged, preserving existing remote resume identities.

For navigation baselines with the same **12 actual moves** as the models, run
each online search separately:

```bash
.venv/bin/python scripts/benchmark_bfs.py --algorithm bfs --max-steps 12 --training-config configs/snap-full.toml --splits dev test --output runs/bfs-12-snap-full.json
.venv/bin/python scripts/benchmark_bfs.py --algorithm dfs --max-steps 12 --training-config configs/snap-full.toml --splits dev test --output runs/dfs-12-snap-full.json
```

These modes expose outgoing links only after visiting a page. Every directed
edge traversal costs one move, including revisits and travel between branches;
there is no teleport or browser-back action. A visible target link is always
taken immediately. BFS prioritizes the oldest discovered frontier page; DFS
prioritizes the newest branch, trying the first source link first. They travel
to the first reachable frontier page along a shortest route using only links
already revealed. Frontier pages that cannot currently be reached stay pending.
An episode fails if no frontier can be reached or the move limit is exhausted.
Neighbor order is deterministic and follows the dataset. These are online
navigation adaptations of BFS/DFS, with no target-distance or text heuristics.

Reports include success rate across all selected pairs, failure reasons, actual
move traces and backtracking counts, total wall time, mean time per task, and
median/p95 episode latency. `--max-steps` overrides only this offline script;
otherwise it uses the selected config's click limit. Omitting `--algorithm`
preserves the original full-graph BFS planner, also available as
`--algorithm full-bfs`. All modes leave remote evaluation settings and resume
identities unchanged.

The real encoder parity tests use the preserved local tokenizer and installed
optional dependencies; a minimal environment skips those checks. Existing
HTTP tests require permission to bind localhost sockets. See
[the implementation and pilot report](docs/jev-results.md) for measured
results, protocol differences, limitations, and provenance.

### Full-split background execution with a shared budget

Run all 179 validation and 175 test pairs per model as a durable background job:

```bash
.venv/bin/python -m wiki_speedrun.batch --background
```

This runs MiniMax, DeepSeek, and Jev, in that order, along with the three pinned
local checkpoints, and caps **combined reported spend at $9** across all splits, probes,
retries, and interrupted episodes. The existing foreground `benchmark` command
also now has the same default shared cap; change it explicitly with
`--max-total-cost`. The final request may overshoot slightly. A response lacking
cost stops the suite until charges can be reconciled. Previously reported
charges are reconstructed from traces when the job restarts.

The job writes `runs/openrouter-paid-full/job.json`, `budget.json`, and `worker.log`.
It resumes temporary network/rate-limit failures after five minutes and daily
quota failures after an hour. It stops on permanent API/configuration errors
or budget limits. It does not retry model-authored invalid decisions: those
remain completed benchmark failures. Repeat the background command after a
machine restart to resume; invoking it while the worker is active does not
start another worker. No task sampling is available in the background runner.

The full report is written to `docs/jev-paid-full-results.md` only after all intended
episodes pass coverage and artifact validation. Full raw results and per-split
paired statistics are under `runs/openrouter-paid-full/`. The five-task report stays
available as implementation history; it is not substituted for full evaluation.

The paid lineup starts in new output directories because the earlier suite's
model list and source identity are frozen for resume. Earlier Mammoth/Elephant
artifacts remain in `runs/openrouter-full/`; its offline report script still
rebuilds the historical report at `docs/jev-full-results.md`.
