# Plan: trainable Kev navigation with shorter-path rewards

Status: implementation plan, October 21, 2026. Implement with Luna agents as
described in the final section, with the coordinating agent reviewing and
integrating their code.

The implementation is now available. See [the training guide](training.md) for
current commands and [the validation report](training-validation.md) for actual
checks. One operational adjustment bounds training development evaluation to a
fixed seeded sample of 16 tasks every 100 updates by default; standalone
evaluation still supports the full split. Resolved settings are in
`configs/training.toml`.

Personal-computer defaults now limit preparation to 2,000 paths, SFT to 200
updates or one epoch, and RL to 20 updates with four episodes each. Checkpoint
retention keeps the two newest committed checkpoints and the best per run.
The original example budgets below describe the initial plan; use the current
TOML and training guide for actual defaults.

## Decisions for the first implementation

- Use Python 3.12, the released `jaredpalmer/kev-0.8b@v1.0` checkpoint, and the
  already-pinned Kev source revision `6b719c3c3f367295f6ef336f4f751cf5ff970abc`.
- Continue training the existing LoRA adapter and pointer head. Freeze the base
  model. Start with SFT on cleaned human routes, followed by on-policy REINFORCE.
- Resolve large action sets with a tournament of Kev predictions. Every outgoing
  link enters the tournament. Start with 64 choices per prediction, subject to
  actual token budgets; keep 255 as the hard upstream API ceiling.
- Reward success, penalize each actual navigation click, and penalize terminal
  failures. Internal tournament predictions are computation, not navigation clicks.
- Keep the HTTP baseline working. Train and evaluate new checkpoints in process
  through the same observation builder, tokenizer, and tournament logic.
- Use ordinary Python modules, dataclasses, PyTorch, TOML, JSON, and JSONL. Do not
  add an RL framework, distributed workers, a value network, or PPO for version 1.

## 1. Python and WSL setup

Python 3.12.15 is already installed through uv on this machine. Do not use the
system Python 3.14 or copy Windows virtual environments into Linux. The Linux
setup script will select Python explicitly and create a fresh `.venv`:

```bash
# Install only if a compatible Python is missing.
uv python install 3.12

# Baseline application; valid with today's repository.
uv sync --locked --python 3.12
.venv/bin/python --version
```

During implementation, add a `train` optional dependency group to `pyproject.toml`,
an exact Git source for Kev, and regenerate `uv.lock` under Python 3.12. Resolve
one Linux environment containing both this application and pinned upstream Kev.
Pin CUDA PyTorch to the baseline's `2.8.0+cu128` using its dedicated uv package
index; do not redirect all dependencies to the PyTorch index. Lock the compatible
PEFT, Transformers, and remaining transitive versions. The new setup command is:

```bash
# Proposed script; implement in phase 1.
bash scripts/setup_training.sh
# Internally: uv sync --locked --python 3.12 --extra train
.venv/bin/python scripts/check_gpu.py
```

Keep `HF_HOME` and `UV_CACHE_DIR` inside `.cache/`. Setup must fail clearly if an
incompatible existing environment is found, verify the Kev revision, and perform
a CUDA operation. Start with one GPU, fp32 model/adapter storage and bf16 autocast
on CUDA, matching upstream training's approach. Measure peak memory before
increasing microbatch size. Serving-only fused kernels and CUDA graphs stay off
for training. Record the resolved Hugging Face commit as well as the `v1.0` tag.

## 2. Modules and interfaces

| File | Responsibility |
| --- | --- |
| `scripts/setup_training.sh` | Reproducible Linux environment and GPU check |
| `configs/training.toml` | Model, data, context, SFT, RL, rewards, persistence |
| `src/wiki_speedrun/models/trainable_kev.py` | Load trainable checkpoint; encode requests; return logits; export weights |
| `src/wiki_speedrun/choices.py` | Ordered candidates, token-safe grouping, tournament execution and replay |
| `src/wiki_speedrun/data/training.py` | Validate human routes, split pairs, build SFT examples and RL task files |
| `src/wiki_speedrun/training/config.py` | Parse/validate training TOML without changing baseline config semantics |
| `src/wiki_speedrun/training/sft.py` | Supervised cross-entropy and development scoring |
| `src/wiki_speedrun/training/rollouts.py` | Collect sampled episodes and recompute stored node log probabilities |
| `src/wiki_speedrun/training/rewards.py` | Pure reward and return calculations |
| `src/wiki_speedrun/training/rl.py` | On-policy batches, advantages, optimizer updates and development evaluation |
| `src/wiki_speedrun/training/checkpoints.py` | Atomic export/resume and run metadata |
| `src/wiki_speedrun/cli.py` | Thin command dispatch; preserve existing commands |

Add `training/__init__.py`. Extend `observations.py` to build the same state with
an explicit ordered subset of outgoing candidates. Keep metadata, labels, graph
distances, and rewards outside model inputs. Keep the baseline's original
`option_limit` behavior selectable for exact historical comparisons.

The wrapper's small public interface should be:

```python
TrainableKev.from_checkpoint(checkpoint, device, context)
model.encode(request)                 # strict encoded input and ordered choice IDs
model.logits(encoded)                 # differentiable float32 tensor [num_choices]
model.save_checkpoint(directory)      # upstream-compatible adapter + head export
```

Sampling and tournaments belong outside the model wrapper. The normal policy
returns the final article plus an explicit original link ID and node trace.
`NavigationEnvironment.step` still runs exactly once per selected article.

### Verified upstream integration points

The pinned upstream has training code that this project can reuse:

- Construct `DecisionModel` from checkpoint `Meta` and adapter configuration,
  then call `Checkpoint.warm_start(model, meta)`. Match base revision, LoRA rank,
  target modules, head size, and special-token settings; never initialize a new
  random adapter in place of released Kev weights. Assert that the trainable
  parameter list contains adapter/head tensors and excludes base tensors.
- Convert requests with `SystemOneRequest` and `api.to_record`, preserving the
  returned choice-key ordering. Use `model.encode(..., strict=True, ...)` with
  explicit context limits, then differentiable `DecisionModel.forward(enc)[0]`.
- Do not train through `probs`, `LocalPredictor`, HTTP output, serving prefix
  caches, or CUDA graph helpers: their inference paths disable gradients.
- Upstream's ordinary checkpoint loader defaults to merged adapters, and PEFT's
  inference loading freezes adapters. The warm-start training path avoids both.
- Upstream applies fitted temperature in eval mode. Use raw logits with head
  temperature `1.0` throughout training/rollout/replay and export this value for
  new checkpoints. Compare to upstream with temperature also set to `1.0`.
- In RL, use eval mode to disable dropout while retaining trainable parameters.
  `eval()` does not disable autograd. Rollout uses `torch.no_grad()`; replay uses
  the identical forward path with gradients enabled. This avoids stochastic
  dropout changing the probabilities between sampling and replay.

These details were checked against pinned upstream
[model.py](https://github.com/jaredpalmer/kev/blob/6b719c3c3f367295f6ef336f4f751cf5ff970abc/kev/model.py),
[checkpoint.py](https://github.com/jaredpalmer/kev/blob/6b719c3c3f367295f6ef336f4f751cf5ff970abc/kev/checkpoint.py),
[api.py](https://github.com/jaredpalmer/kev/blob/6b719c3c3f367295f6ef336f4f751cf5ff970abc/kev/api.py),
and [train.py](https://github.com/jaredpalmer/kev/blob/6b719c3c3f367295f6ef336f4f751cf5ff970abc/kev/train.py).

## 3. Many outgoing links: a tournament

For each navigation click:

1. Give every outgoing edge its original `link_<index>` ID. Keep source order,
   duplicates, self-links, and visited links; there is no automatic target shortcut.
2. Partition candidates into consecutive groups of at most 64 choices. Strictly
   encode each proposed group with the real tokenizer, accounting for state,
   instruction, option text, and special tokens. Split oversized groups further.
3. Ask Kev to select one winner per group. RL samples from a categorical
   distribution over raw logits; deterministic evaluation takes argmax.
4. Feed the winners' original IDs and titles into the next level. Repeat grouping
   and prediction until only one winner remains. Each round must reduce the
   candidate count; pass singleton groups through without a model call.
5. Click the final article and compute one environment reward.

`United_States` has 294 outgoing links: with 64-choice groups that fit the token
budget, five initial predictions produce five winners, followed by one final
prediction. The result is one navigation click and six model calls. A smaller
token budget can increase the number of calls. Log both counts independently.

Use identical state/instructions at each level and retain original link IDs
instead of renumbering winners. The default grouping is deterministic and
independent of the target and model scores. Log round, node, candidates, request,
chosen ID, and sampling log probability. Uneven groups can introduce tournament
bias; measure sensitivity to seeded candidate permutations as an ablation, not
as an unrecorded training-only behavior.

### Token budgets and guaranteed progress

Initial explicit limits: 1,024 state tokens, 4,096 tokens for the full state plus
one question row. Keep the current 1,500-character excerpt cap as an initial
upper bound, but enforce limits using tokenizer output. Preserve current/target
titles; remove excerpt tokens first and then the oldest history entries to fit
the state budget. Log any state shortening. Never silently omit a link.

Preflight must verify that every pair of candidate descriptions on this fixed
graph can fit with the bounded state/instruction, ensuring a level can reduce.
If necessary, shorten display titles deterministically while preserving full
IDs in metadata and logging the transformation. If even two candidates cannot
fit, raise a configuration error and stop before training. A runtime overflow
or CUDA exception is an infrastructure failure, not a terminal navigation reward.

### Why this choice

| Approach | Role in this plan |
| --- | --- |
| Pre-rank and keep top K | Optional later speed experiment; permanently removes choices and requires recall checks |
| Group selection then link selection | Optional later alternative; fewer calls but requires useful descriptions of groups |
| Chunk winners followed by another Kev prediction | Default; covers every edge and directly implements predictions feeding later predictions |

Do not compare independent chunk logits as if they shared a global scale. The
final Kev prediction compares the surviving candidates in one context. The
tournament may eliminate a useful link early; SFT and RL must train those early
predictions too. Version 1 passes winners rather than generating text rationales.

## 4. Data preparation and SFT warm start

Use completed human paths only. Explicitly treat `load_paths` output as cleaned
forward routes after removing backtracking, not chronological human behavior.
Validate every remaining adjacent edge. Drop the entire route on an illegal
edge; do not join across removed edges or silently invent a shortcut. Log the
six known invalid transitions and all rejection counts. Keep routes with 1–12
forward clicks initially; save exclusion counts and source row numbers.

Assign complete `(start, target)` pairs to train/dev/test before expanding any
route into examples. Use sorted canonical article IDs and a seeded SHA-256
assignment with 80/10/10 buckets; all demonstrations for the same pair share a
split. Reserve the existing ten baseline pairs for development. Persist actual
assignments; subsequent commands load the manifest rather than reshuffling.
The split guarantees pair disjointness, not disjoint articles or intermediate
states. The policy is expected to navigate one shared graph.

For each route step, build the same history/state as rollout would expose. If a
tournament is necessary, ensure the demonstrated next link survives its group
at every level and create cross-entropy labels for its node at each level.
Choose other groups' winners with a seeded uniform rule during preprocessing;
record them, and do not put supervised loss on arbitrary non-target winners.
No demonstrated label or forced winner is available at inference. Retain every
node's actual candidates, so the exported examples fully specify the training input.

Use one microbatch at a time, AdamW, gradient accumulation, and gradient clipping.
Train existing adapter and head at an initial learning rate of `2e-5`, selecting
the epoch using development navigation success and then successful path length.
An initial maximum of two epochs is a starting budget, not a claim of convergence.

If sparse successes prevent RL learning, first add explicitly labelled synthetic
shortest routes for training-assigned pairs or use an easier distance curriculum.
Record `source = "shortest_path"` separately from `"human"`. Graph search may
construct training labels, but distances/routes never enter policy observations.
All new task pairs go through the same persisted split assignment.

## 5. Reward: successful and shorter paths

Start with three transparent numbers:

```text
each actual click:                  -0.02
reaching target on that click:      +1.00 additionally
dead end or 12-click limit:         -1.00 additionally
```

Use undiscounted returns (`gamma = 1.0`). Success takes precedence over the click
limit and dead-end checks when the final article is the target. With `T` clicks,
successful return is `1 - 0.02*T`; failed return is `-1 - 0.02*T`. A two-click
success returns `0.96`, a ten-click success `0.80`, and a twelve-click failure
`-1.24`. Thus every allowed success beats every failure, and fewer clicks improve
the return among successes. Validate `step_cost * max_steps < 2` when configuring
rewards. There is no voluntary stop action, revisit bonus, graph-distance reward,
or reward for internal Kev predictions in version 1.

An episode starting in a non-target dead end receives terminal `-1` with zero
actions; record it but it contributes no policy-gradient term. Exclude such starts
and `start == target` from generated training tasks. An infrastructure exception
aborts the batch and produces an error artifact, not a fictitious episode reward.

Report success rate first, then successful clicks and shortest-path overhead.
Mean clicks across failures can misleadingly improve if the policy dies early.

## 6. RL loop and tournament credit assignment

Use small on-policy REINFORCE batches after SFT. For each update:

1. Choose four training task pairs and sample four independent episodes per pair
   with the current fixed policy (16 episodes). Cap each at 12 navigation clicks.
2. Collect without gradients; store every tournament node, its exact request,
   chosen ID, float32 log probability, and policy version. Keep the model fixed
   until collection and gradient accumulation for the whole batch finish.
3. Compute reward-to-go for each navigation click. For each episode, use the mean
   total return of the other three episodes for its pair as a detached baseline.
   Subtract it from each click's reward-to-go. This baseline does not depend on
   the episode's own sampled choices. No value network is needed.
4. Replay each stored node under the unchanged policy with gradients enabled,
   verify its chosen log probability matches collection within a stated numeric
   tolerance, and backpropagate one node/microbatch at a time. Accumulate for the
   full batch; normalize the policy loss by episode count, not variable path length.
5. Clip gradients at `1.0`, take one AdamW step (`5e-6` initial LR), and discard
   the batch. Never reuse it for multiple optimizer updates in version 1.
6. Append metrics, periodically evaluate fixed development tasks, and checkpoint
   at completed optimizer boundaries. Select best by development success, then
   successful clicks; do not select on test performance.

For click `t`, define the joint internal decision log probability:

```text
log_pi_t = sum(log p(selected_node_choice | that_node_request))
           over EVERY sampled node used to resolve this click

policy_loss = -sum((return_to_go_t - baseline_episode) * log_pi_t)
              / number_of_episodes
```

Include predictions in groups whose winners later lose. They changed the final
comparison and are part of the joint stochastic computation. Training only the
final prediction, or multiplying probabilities only along the final winner's
ancestry, is incorrect for this tournament. `log_pi_t` is the probability of the
recorded internal trace, not the marginal probability of the final article.
Singleton pass-throughs have no sampled decision and no gradient term.

Start without an entropy bonus (`0.0`) to keep the objective exact and readable;
categorical sampling provides exploration. Log entropy so a later explicit
experiment can add a small bonus if the policy collapses. Keep dropout off and
temperature `1.0` in both collection and replay. Deterministic argmax evaluation
is a separate mode. Avoid keeping autograd graphs for whole episodes in memory.

## 7. Specific file formats

All text files use UTF-8 and LF line endings. JSON objects have `schema_version`
where the format is new. Article IDs retain underscores and decoded Unicode.
JSONL contains one complete JSON object per line and never NaN/Infinity. Pretty
examples below illustrate records; serialize each JSONL record on a single line.
Treat option arrays as ordered; do not depend on re-sorting JSON object keys.

### Configuration: `configs/training.toml`

Keep baseline TOML separate. Example proposed defaults:

```toml
schema_version = 1
seed = 42
data_dir = "data"
run_dir = "runs"

[model]
checkpoint = "jaredpalmer/kev-0.8b@v1.0"
kev_revision = "6b719c3c3f367295f6ef336f4f751cf5ff970abc"
device = "cuda"
weights_dtype = "fp32"
autocast_dtype = "bf16"
temperature = 1.0

[choices]
strategy = "tournament"
node_max_choices = 64
max_state_tokens = 1024
max_row_tokens = 4096
excerpt_chars = 1500
grouping = "source_order"

[data]
split_manifest = "data/splits/training/manifest.json"
sft_train = "data/training/sft_train.jsonl"
sft_dev = "data/training/sft_dev.jsonl"
rl_train_tasks = "data/splits/training/train_tasks.json"
dev_tasks = "data/splits/training/dev_tasks.json"
test_tasks = "data/splits/training/test_tasks.json"

[sft]
epochs = 2
learning_rate = 0.00002
microbatch_size = 1
gradient_accumulation = 8

[rl]
max_steps = 12
updates = 100
task_pairs_per_update = 4
episodes_per_pair = 4
learning_rate = 0.000005
gamma = 1.0
entropy_coefficient = 0.0
max_grad_norm = 1.0

[reward]
success = 1.0
failure = -1.0
step_cost = 0.02

[save]
checkpoint_every_updates = 10
evaluate_every_updates = 10
```

Reject unknown settings, invalid bounds, fewer than two episodes per pair, a
choice limit outside `2..255`, or configuration/data mismatches on resume.

### Task splits: JSON arrays plus a manifest

`data/splits/training/{train,dev,test}_tasks.json` keeps the existing compatible
task format:

```json
[{"start": "Earth", "target": "Moon"}]
```

`manifest.json` records schema version, seed, split fractions, SHA-256 hashes of
raw path input, processed graph, baseline task file, each task file, and SFT
files; pair assignments; source row IDs; preprocessing version; choice/token
configuration; tokenizer/base commit; rejection counts and reasons. Represent
pair assignments as an array of `{start, target, split}` objects, avoiding
ambiguous delimiter-based pair keys. Include task provenance and shortest
distances as metadata only. A deterministic rerun must produce identical files.

### SFT examples: `data/training/sft_{train,dev}.jsonl`

One record per supervised tournament node. Example structure (link numbering is
illustrative):

```json
{
  "schema_version": 1,
  "example_id": "human-000123-step-0-round-0-node-0",
  "trajectory_id": "human-000123",
  "pair": {"start": "Earth", "target": "Moon"},
  "source": "human",
  "step": 0,
  "round": 0,
  "node": 0,
  "request": {
    "model": "kev-latest",
    "state": {
      "current_title": "Earth",
      "target_title": "Moon",
      "current_excerpt": "Earth is a planet.",
      "history": ["Earth"]
    },
    "questions": {
      "next_link": {
        "type": "choice",
        "instructions": "Choose the outgoing link that best advances toward the target.",
        "criteria": {"link_0": "Moon", "link_1": "Sun"}
      }
    }
  },
  "candidate_ids": ["link_0", "link_1"],
  "articles": ["Moon", "Sun"],
  "label": "link_0"
}
```

Only `request` enters model serialization. `label` resolves through ordered
`candidate_ids`; validate exact correspondence with request criteria. Node
construction must use one shared instruction template in SFT and inference.
The manifest fingerprints that template and the observation preprocessing.

### Run artifacts: JSON and streaming JSONL

Create `runs/<UTC-stamp>-{sft,rl}/` containing:

| File | Contents |
| --- | --- |
| `config.json` | Fully resolved configuration, schema version, run ID |
| `provenance.json` | Local Git revision/dirty state, source/split hashes, model commits, package versions, GPU, preprocessing versions |
| `metrics.jsonl` | One record per completed update/evaluation: loss, gradient norm, returns, success, clicks, node calls, entropy, elapsed time, peak memory |
| `transitions.jsonl` | One record per completed environment click, including all internal node decisions |
| `episodes.jsonl` | One record per completed episode with path, return, success, termination reason, policy version |
| `metrics.json` | Latest aggregate evaluation metrics, including failure counts and successful-path overhead |
| `error.json` | On infrastructure failure: error type/message, update/episode IDs, last completed checkpoint |

Each transition contains `schema_version`, `policy_version`, `episode_id`,
`pair`, `step`, `source`, `target`, `history`, `selected_link_id`,
`selected_article`, `reward`, `next_article`, `termination_reason`, and `nodes`.
Each node contains `round`, `node`, its exact `request`, ordered `candidate_ids`,
parallel `articles`, `selected_link_id`, `old_log_prob`, and `entropy`. Record
token counts and state-shortening metadata. Detached numeric fields are audit
data; gradient replay reconstructs logits from requests, not saved probabilities.

Flush after each completed click/episode/update. A trailing partial JSONL line
from an abrupt stop may be discarded on recovery; do not discard earlier lines.
Use explicit failure categories: success, step limit, dead end, and infrastructure
error. The tournament policy must not produce `option_limit` on the validated
graph; historical baseline output may still use that category.

### Checkpoints: upstream weights plus training state

```text
checkpoints/<run-id>/update-000010/
  adapter_config.json
  adapter_model.safetensors
  head.pt
  tokenizer_config.json
  tokenizer.json                 # plus tokenizer files required by upstream
  training_state.pt
  manifest.json
checkpoints/<run-id>/latest.json
checkpoints/<run-id>/best.json
```

Export the adapter with PEFT `save_pretrained(..., safe_serialization=True)` and
the pointer head plus metadata through upstream `Meta` / `write_meta`. Preserve
base/base revision, LoRA/head architecture, token settings, weight dtype, and
temperature. `head.pt` follows upstream's PyTorch dictionary format; do not
invent an incompatible weights format. Base model weights are referenced by
immutable commit and cached rather than duplicated into every checkpoint.

`training_state.pt` is a local PyTorch state dictionary holding optimizer and
scheduler state (if used), completed update, epoch/data cursor or next task
sampler state, Python/NumPy/Torch CPU/CUDA RNG states, best development score,
and run ID. Persist resume at completed optimizer boundaries only, with no
pending gradients or half-consumed rollout batch. `manifest.json` holds format
version, phase (`sft`/`rl`), file hashes, full model/config/data fingerprints,
observation/tournament versions, and committed JSONL offsets.

Write a temporary checkpoint directory on the same filesystem, finish files and
hashes, rename atomically, then atomically replace `latest.json` with
`{"schema_version":1,"checkpoint":"update-000010"}`. Never point at partial
exports. Resume validates fingerprints, restores all state, ignores uncommitted
rollouts after the saved boundary, and truncates logs to committed offsets before
continuing. Test the next sample/update against an uninterrupted run, allowing
documented CUDA numeric tolerance. Keep `best.json` separate from latest.

## 8. Implementation phases and acceptance checks

| Phase | Deliverable | Required evidence before moving on |
| --- | --- | --- |
| 1 | Python/CUDA setup and trainable wrapper | GPU matmul; strict token admission; logits agree with upstream at matched precision and temperature; gradients reach only intended parameters |
| 2 | Tournament and versioned datasets | 294-link page resolves legally; every candidate covered; recursion reduces; no dropped labels; deterministic files; disjoint task pairs; no graph hints in input |
| 3 | SFT and checkpoint round trip | Tiny fixed batch loss decreases; adapter/head weights change, base stays fixed; reload preserves logits; resumed step matches uninterrupted step |
| 4 | RL loop | Reward arithmetic; shorter success has larger return; every sampled tournament node gets credit; replay log-probability agreement; finite gradient/update; resume preserves sampling |
| 5 | Short measured run | Memory and runtime measured; untuned and SFT comparisons completed; artifacts survive interruption; no silent invalid actions or infrastructure failures |

Build small deterministic tests for reward/termination boundaries and tournament
credit assignment, plus real-GPU integration checks for autograd and reload.
Keep existing 17 tests passing. Test success on the last allowed click and at a
target with zero outgoing edges, singleton groups, duplicate edges, token-driven
splitting, and a forced two-level tournament where losing groups still affect
the final comparison. Verify all internal log probabilities enter the loss.

Evaluate three policies on identical tasks and preprocessing: released Kev with
the original baseline settings, released Kev with the new tournament settings,
and trained Kev with those same tournament settings. This separates preprocessing
effects from learning. Keep the recorded ten tasks, add a fixed larger development
set and a separate large-link stress set, and reserve test evaluation until the
training approach and checkpoint are selected. Report counts and uncertainty;
ten pairs alone cannot establish an improvement.

## 9. Proposed commands after implementation

```bash
bash scripts/setup_training.sh
.venv/bin/python -m wiki_speedrun prepare-training --config configs/training.toml
.venv/bin/python -m wiki_speedrun train-sft --config configs/training.toml
.venv/bin/python -m wiki_speedrun train-rl --config configs/training.toml --init-from checkpoints/<sft-run>/best-checkpoint
.venv/bin/python -m wiki_speedrun train-rl --config configs/training.toml --resume checkpoints/<rl-run>/update-000010
.venv/bin/python -m wiki_speedrun evaluate-trained --config configs/training.toml --checkpoint checkpoints/<rl-run>/update-000010 --split dev
```

These are new subcommands; add training `--config` arguments to those subparsers
without changing existing baseline commands' global `--config` behavior. Resolve
`best.json` to an actual checkpoint path; `best-checkpoint` above is a placeholder.
Initial implementation validation uses tiny explicit step/task limits, then a
short run. Choose a longer budget from measured runtime and GPU memory.

## 10. Luna implementation assignments and code quality

When implementing, use `gpt-6-luna` agents with explicit bounded tasks and the
interfaces/formats in this plan. Pass a self-contained brief (model override
requires a fresh-context agent), relevant file contents/paths, owned files, and
acceptance criteria. The coordinator owns integration, reviews every diff, and
runs the required checks. Do not hand off the whole implementation without review.

Start by freezing shared dataclasses, request formatting, and checkpoint contracts.
Then use at most three agents concurrently with non-overlapping ownership:

| Luna assignment | Owned work | Dependency |
| --- | --- | --- |
| Environment and model | Linux setup, training dependencies/lock, wrapper, wrapper integration checks | Shared request/checkpoint contract |
| Choices and data | Tournament, observation subsets, pair splits, JSONL examples, data/choice tests | Shared types and tokenizer interface; can start with a fake scorer |
| Learning loops | Rewards, rollout/replay, SFT/RL optimizers, persistence and focused tests | Reward tests first; full loops after wrapper and choices integrate |

The coordinator integrates CLI/config/README changes after module interfaces
stabilize. Reassign narrow follow-ups to Luna agents for integration fixes. Review
math and end-to-end behavior centrally before launching a real training job.

Include this instruction in every Luna brief:

> Write clean, human-readable Python with descriptive names, short functions,
> explicit data flow, type hints on public interfaces, and brief comments for
> the non-obvious math. Prefer dataclasses and ordinary loops. Avoid clever
> comprehensions, plugin registries, generic trainer frameworks, duplicated
> serialization, broad exception swallowing, and silent fallbacks. Implement
> only your assigned scope. Keep the existing baseline behavior intact. Add
> focused tests for algorithmic correctness and report assumptions and checks.

Coordinator review must check gradients, adapter loading, tournament coverage,
all-node credit assignment, reward boundaries, split leakage, exact observation
replay, atomic resume, and baseline compatibility. Simplify unnecessary abstractions
before accepting a change. Keep training loops readable from top to bottom.
