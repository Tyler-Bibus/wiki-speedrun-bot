# Supervised model evaluation

Evaluated October 22, 2026 on the local NVIDIA GeForce RTX 3090 (24 GiB).
This evaluation uses the full frozen validation (`dev`) and test task splits
from the prepared 2,000-human-path sample. Released Kev is evaluated on the same
validation split to measure the improvement from Wikispeedia SFT.

## Results

| Metric | Untuned validation (`dev`) | SFT validation (`dev`) | SFT test |
| --- | ---: | ---: | ---: |
| Tasks evaluated | 179 | 179 | 175 |
| Successful tasks | **22/179 (12.29%)** | **162/179 (90.50%)** | **157/175 (89.71%)** |
| Mean clicks, all episodes | 11.25 | 4.83 | 4.78 |
| Mean clicks, successful episodes | 5.91 | 4.07 | 3.96 |
| Mean shortest-path clicks, successful episodes | 2.55 | 2.79 | 2.78 |
| Mean excess clicks over shortest path, successful episodes | 3.36 | 1.28 | 1.17 |
| Click-limit failures | 157 | 17 | 18 |
| Dead-end failures | 0 | 0 | 0 |
| Total revisits | 1,082 | 102 | 119 |
| Mean revisits per episode | 6.04 | 0.57 | 0.68 |
| Total model calls | 3,347 | 1,680 | 1,676 |
| Mean model calls per episode | 18.70 | 9.39 | 9.58 |
| Mean navigation-decision latency | 355.6 ms | 468.7 ms | 484.1 ms |
| Evaluation wall time, excluding model loading | ~727.4 s | 411.5 s | 411.9 s |

Navigation-decision latency includes all tournament calls needed for one click;
it is not latency per internal model call. The SFT validation and test evaluations took
approximately 13.7 minutes in total, excluding initial model loading and checks.
Untuned validation took approximately 12.1 minutes. Its wall time is estimated
from run artifact timestamps because an extra metadata step failed after all
episodes and metrics were saved; the metadata was recovered without rerunning
evaluation. There were no navigation-evaluation errors or omitted tasks.

The 16-task selection subset reproduced **16/16 successes and 3.75 mean clicks**
in the full validation run. The other 163 validation pairs achieved
**146/163 successes (89.57%)**, with 4.11 mean successful clicks. The test success
rate is close to this validation result outside the selection sample.

All SFT failures reached the 12-click limit. Repeated pages occurred in
16/17 validation failures and 18/18 test failures. For example,
`Abuja -> Internet_Explorer` reached `Microsoft`, then repeatedly alternated
between `Microsoft` and `Microsoft_Windows` until the limit. Full paths and
individual tournament decisions are saved below.

These results describe the selected SFT checkpoint. The final checkpoint was
not evaluated on the full splits, so this does not measure the full-set effect
of training beyond update 1,700. The matched validation comparison below uses
the same protocol for released and SFT weights. The historical ten-task HTTP
baseline remains a separate experiment with different settings.

## Matched validation comparison

SFT improved success from **22/179 (12.29%) to 162/179 (90.50%)**, a gain of
**78.21 percentage points** and **140 additional solved tasks**.
Both runs use exactly the same validation pairs, graph, observation formatting,
tournaments, precision, and click limit. Only the adapter and pointer-head
weights differ.

| Paired outcome on the same 179 validation tasks | Tasks |
| --- | ---: |
| Both models solved | 22 |
| SFT solved, untuned failed | 140 |
| Untuned solved, SFT failed | 0 |
| Both failed | 17 |

On the **22 tasks both models solved**, mean clicks fell from
**5.91 to 3.05**. This compares route efficiency on an identical task subset;
the successful-episode means in the main table cover different subsets.
All 157 untuned failures reached the click limit. Total validation revisits fell
from 1,082 to 102, and internal model calls fell from 3,347 to 1,680.

Excluding the 16 development pairs used for checkpoint selection, untuned Kev
solved **20/163 (12.27%)**, versus **146/163 (89.57%)** for SFT. Thus the gain
also appears on validation pairs outside the checkpoint-selection sample.

Here, untuned means the released `jaredpalmer/kev-0.8b@v1.0` adapter and pointer
head, before additional Wikispeedia training. The evaluated release revision is
`bf75a6a8848ea6960ff2ed108d9ed44c2941174f`; its adapter and head hashes match
the original SFT initialization. It is not the bare Qwen base model. No untuned
test evaluation was run for this comparison.

## Checkpoint selection

The overnight SFT run `20261022T025733.517853Z-sft` completed 2,000 optimizer
updates. Its saved `best.json` points to **`update-001700`**, which was preserved
alongside updates 1,950 and 2,000. Selection uses the highest navigation success
rate on a fixed, seeded sample of 16 development pairs, breaking ties by fewer
clicks among successful episodes.

Update 1,700 solved 16/16 selection tasks with 3.75 mean successful clicks.
The final update solved 15/16 with 3.87 mean successful clicks. Update 600 also
solved 16/16, but averaged 4.00 clicks, so update 1,700 correctly replaced it.
The selected checkpoint was fixed before evaluating the full validation or test
split; these results do not change the saved best pointer.

The later 15/16 results show a small drop on the selection sample, but do not
by themselves establish overfitting. The sample is small and was reused during
training. There is no logged held-out supervised-loss curve: training selected
checkpoints using navigation outcomes.

## Protocol

- Use [configs/overnight.toml](../configs/overnight.toml) and the existing
  prepared splits: 1,404 training, 179 validation, and 175 test start/target pairs.
  Evaluate every validation and test pair once, without a task limit.
- Verify all checkpoint-file hashes, the original configuration fingerprint,
  the prepared training/development file hashes, and the graph fingerprint
  against the selected checkpoint manifest before loading weights.
- Verify task pairs are unique and disjoint across train/dev/test, and that
  supervised decision records belong to their declared pair split. These are
  held-out pairs on the same article graph; articles can occur in multiple splits.
- Use deterministic tournament argmax, 64 choices per group, source ordering,
  a 12-click limit, 1,500-character excerpts, 1,024 state tokens, and a 4,096-token
  row budget. All outgoing links participate. No target shortcut, graph-distance
  hints, or visited-page filtering is applied to the policy.
- Load the selected LoRA adapter and pointer head over the frozen
  `Qwen/Qwen3.5-0.8B-Base` revision
  `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`. Use fp32 weights with bf16
  autocast, the same reference PyTorch kernels as training, and no gradients.
- Use Python 3.12.15, PyTorch 2.8.0+cu128, Transformers 5.20.2, PEFT 0.21.2,
  and Kev source revision `6b719c3c3f367295f6ef336f4f751cf5ff970abc`.

These are complete navigation rollouts from each held-out start to its target.
They measure task success and route efficiency, rather than teacher-forced
accuracy on individual human decisions. This is the complete prepared test
split for the pilot sample, not all human paths in the SNAP dataset.

## Reproduction

From the repository root, with the existing prepared dataset and checkpoints:

```bash
.venv/bin/python -m wiki_speedrun evaluate-trained \
  --config configs/overnight.toml \
  --checkpoint checkpoints/20261022T025733.517853Z-sft/update-001700 \
  --split dev
.venv/bin/python -m wiki_speedrun evaluate-trained \
  --config configs/overnight.toml \
  --checkpoint checkpoints/20261022T025733.517853Z-sft/update-001700 \
  --split test
```

Run the released checkpoint on the same validation split:

```bash
.venv/bin/python -m wiki_speedrun evaluate-trained \
  --config configs/overnight.toml \
  --checkpoint jaredpalmer/kev-0.8b@bf75a6a8848ea6960ff2ed108d9ed44c2941174f \
  --split dev
```

The recorded untuned run loaded that exact revision from the local cache.
The recorded SFT evaluation reused one loaded model and graph for the two calls to
`evaluate_trained`, with the same settings as these commands. Each split writes
its own resolved configuration, provenance, aggregate metrics, episodes, and
incremental tournament decision traces under `runs/`.

## Saved artifacts and verification

The following raw artifacts remain local and ignored by Git. This document
records the aggregate results in the repository.

- Untuned validation: [metrics](../runs/20261022T124949.434683Z-evaluate/metrics.json),
  [episodes](../runs/20261022T124949.434683Z-evaluate/episodes.jsonl),
  [tournament traces](../runs/20261022T124949.434683Z-evaluate/transitions.jsonl),
  [provenance](../runs/20261022T124949.434683Z-evaluate/provenance.json),
  [resolved configuration](../runs/20261022T124949.434683Z-evaluate/config.json),
  [benchmark metadata](../runs/20261022T124949.434683Z-evaluate/benchmark.json),
  [verified paired comparison](../runs/20261022T124949.434683Z-evaluate/comparison.json).
- [Evaluation suite summary](../runs/20261022T123257.386086Z-sft-evaluation/summary.json): frozen checkpoint,
  weight hashes, dataset hashes, completeness, metrics, and split wall times.
- [Artifact verification](../runs/20261022T123257.386086Z-sft-evaluation/verification.json): task coverage,
  legal graph moves, aligned transitions, recomputed metrics, and artifact hashes.
- Validation: [metrics](../runs/20261022T123309.717009Z-evaluate/metrics.json),
  [episodes](../runs/20261022T123309.717009Z-evaluate/episodes.jsonl),
  [tournament traces](../runs/20261022T123309.717009Z-evaluate/transitions.jsonl),
  [provenance](../runs/20261022T123309.717009Z-evaluate/provenance.json),
  [resolved configuration](../runs/20261022T123309.717009Z-evaluate/config.json).
- Test: [metrics](../runs/20261022T124001.384841Z-evaluate/metrics.json),
  [episodes](../runs/20261022T124001.384841Z-evaluate/episodes.jsonl),
  [tournament traces](../runs/20261022T124001.384841Z-evaluate/transitions.jsonl),
  [provenance](../runs/20261022T124001.384841Z-evaluate/provenance.json),
  [resolved configuration](../runs/20261022T124001.384841Z-evaluate/config.json).

Verification confirmed exact ordered coverage of all 354 task pairs, unique
episode IDs, all 1,701 navigation transitions matching their episode paths,
legal directed graph edges, and all 3,356 recorded internal model calls.
Aggregate metrics recomputed from saved episodes match the evaluator output.
All prepared output files match the original split manifest, including the
held-out test task list. The original best/latest pointers and training logs
were left intact.

Untuned verification separately confirmed exact ordered coverage of all 179
validation pairs, all 2,014 legal navigation transitions, all 3,347 internal
model calls, matching recomputed metrics, identical evaluation settings and
data hashes, and original initialization weight hashes. Paired comparison
uses the saved episodes from both runs.
