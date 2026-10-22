# Fine-tuning and RL readiness review

Reviewed October 21, 2026, after copying the baseline repository to WSL.

This is the historical review before implementation. The SFT/RL pipeline has
since been added; see the [training guide](training.md) and
[implementation validation](training-validation.md) for its current state.

**Verdict: the baseline works; this checkout cannot train yet.** There are no
fine-tuning or RL entry points, trainable model wrapper, training dependencies,
training configuration, or training checkpoints. The README's “Later training”
section describes future work, rather than scripts already present. No training
job was launched during this review.

## Findings

1. **Launch blocker: training is not implemented.**
   `src/wiki_speedrun/cli.py` exposes data preparation, smoke testing, task
   generation, play, and evaluation only. `models/kev.py` calls an HTTP inference
   endpoint and returns ordinary Python numbers; it has no model parameters or
   gradient path. Implement an in-process Kev adapter/head loader and scorer,
   supervised loss, optimizer, and checkpoint save/load before SFT. RL additionally
   needs sampled legal actions, differentiable action log probabilities, rewards,
   returns/advantages, and a policy update. Existing HTTP episode traces alone
   do not supply these components. Match training's input serialization, choice
   ordering, and scoring to inference, and verify their scores agree before
   measuring improvements.

2. **Launch blocker: human paths are not a validated training dataset.**
   `data/load.py:load_paths` returns the final active stack after backtracking;
   it discards the chronological visits that were undone. The environment has
   forward-link actions only. Choose explicitly between cleaned forward routes
   for SFT and a new action space supporting backtracking. Do not treat the
   current loader as a chronological demonstration reader. On the copied data:

   - 51,318 completed paths cover 28,718 distinct start/target pairs.
   - 8,658 pairs have multiple demonstrations. Split by pair before expanding
     paths into decisions, keeping each complete trajectory in one split.
   - Six adjacent transitions across the final stacks are absent from the graph:
     `Finland -> Åland` (2), `Claude_Monet -> Édouard_Manet` (2),
     `Bird -> Wikipedia_Text_of_the_GNU_Free_Documentation_License` (1), and
     `Republic_of_Ireland -> Éire` (1). Reject or explicitly repair these examples
     and report counts; every SFT label must belong to the submitted legal choices.
   - 8,454 paths contain at least one decision at `United_States`, whose 294
     choices exceed the configured limit of 255. Define a consistent policy for
     training and evaluation before building examples. The current baseline
     ends such episodes with `option_limit`; it does not truncate choices.

3. **Launch blocker: no frozen training/development/test split exists.**
   `make-tasks` samples reachable tasks but does not establish disjoint splits.
   Keep `examples/baseline_tasks.json` as a development comparison, exclude those
   pairs from training, and create a separate held-out test set. None of the ten
   baseline pairs currently occurs as a complete human-path start/target pair.
   Prevent overlap when adding generated routes or RL tasks as well. Keep graph
   search in data construction and metrics, outside policy observations.

4. **Launch blocker: the WSL model environment is not prepared.**
   Neither `.venv` nor `.cache/kev` is present in this copy. The provided setup
   and serving scripts use PowerShell and Windows `Scripts/python.exe` paths.
   Recreate Linux environments and provide Linux launch commands. System Python
   is 3.14.4, outside `pyproject.toml`'s `>=3.12,<3.14` range; Python 3.12.15 is
   already installed through uv. The audit interpreter has no PyTorch. NVIDIA
   inspection outside the sandbox sees an RTX 3090 with 24 GiB, but a PyTorch
   CUDA operation and a model forward/backward pass remain unverified in WSL.

5. **Before a long run: add recovery and reproducibility.**
   The evaluation CLI holds episodes in memory and saves only after the whole
   task list completes. A later HTTP/token-budget failure or interruption loses
   the preceding unsaved results. A training runner needs periodic atomic
   checkpoints, incremental metrics, interruption handling, and resume of model,
   optimizer, scheduler, step, and RNG state. Record the base checkpoint, upstream
   source revision, trainable parameter selection, configuration, and split/data
   fingerprints. Existing baseline metadata is not a training resume mechanism.
   Budget inputs with the actual Kev tokenizer, including history and every
   choice; excerpt character limits alone do not establish token-budget safety.

## Verification completed

- All 17 existing application tests passed with Python 3.12.15. The fake HTTP
  server tests required execution outside the sandbox to bind to loopback.
- The processed graph has 4,604 articles, 119,882 directed links, and text for
  all 4,604 articles. All ten baseline pairs remain reachable at the recorded
  two- or three-click distances.
- The fixed ten-task random evaluation reproduced 0/10 successes, 12 mean
  clicks, and one revisit. Its saved output is
  `runs/20261022T014543013748Z-random/` (UTC timestamp; ignored by Git).
- NVIDIA inspection succeeded outside the sandbox. The sandbox's initial GPU
  access error does not establish a WSL driver problem.
- No real Kev inference, CUDA tensor operation, gradient update, or checkpoint
  round trip was validated in this WSL copy because its model environment is absent.

## Implementation and launch order

1. Prepare Linux Python/CUDA/Kev environments with pinned dependencies and source.
   Run `scripts/check_gpu.py` with the model environment's Linux interpreter.
2. Build and persist pair-disjoint dataset splits and validated SFT decisions.
   Reuse the navigation observation format and record rejected-example counts.
3. Implement SFT from the released Kev weights. Verify nonzero finite gradients
   reach only the intended parameters, parameters change after an optimizer
   step, and loss decreases on a tiny fixed batch. Save/reload and resume a short
   run, checking that inference scores survive the checkpoint round trip.
4. Evaluate the SFT checkpoint against the released model on identical frozen
   development tasks. Measure success, clicks, loops, option-limit failures,
   latency, and memory. Use the separate held-out test set for final assessment.
5. Add RL with explicit success/step/terminal rewards and sampled actions.
   Distinguish success, step limit, dead end, and option-limit termination from
   infrastructure failures; do not train on HTTP/CUDA errors as task failures.
   Check legal-action sampling, finite loss/gradients, reward accounting, actual
   parameter updates, checkpoint reload, and resume on a short run before scaling.

For baseline-only checks in a recreated Linux project environment:

```bash
uv sync --locked --python 3.12
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m wiki_speedrun evaluate --tasks examples/baseline_tasks.json --policy random
```

These commands install and exercise the standard-library baseline application;
they do not install Kev/PyTorch or create a training pipeline.
