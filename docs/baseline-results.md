# First prototype results

Tested October 21, 2026 on the local NVIDIA GeForce RTX 3090 (24 GiB), Python
3.13.3, PyTorch 2.8.0+cu128, and Transformers 5.17.0. The server reported CUDA,
the torch backend, bfloat16 weights, and `jaredpalmer/kev-0.8b@v1.0`. No
fine-tuning, graph-distance hints, target-link shortcut, or visited-link filtering
was used. Settings are in `configs/baseline.toml`.

The model smoke test chose `earth` with probability 0.9876. The one-click
`Earth -> Moon` navigation succeeded, taking approximately 569 ms for its decision.

The fixed [ten-task sample](../examples/baseline_tasks.json) has shortest paths of
two or three clicks. Both policies had a 12-click limit; the random seed was 42.

| Policy | Successes | Average clicks | Revisits | Average decision latency |
| --- | --- | --- | --- | --- |
| Untuned Kev-0.8B | 1/10 | 10.7 | 45 | 278 ms |
| Uniform random | 0/10 | 12.0 | 1 | Not measured |

Kev reached `Anthropology` from `Bushbuck` in 10 clicks, versus a shortest path of
3 clicks. Eight tasks reached the click limit. The remaining task stopped after
entering `United_States`, which has more than 255 outgoing choices. That episode
was included as a failure in the success rate.

The traces show frequent revisits, including repeated self-links. A successful
shortest path of two or three clicks does not mean the model found that route.
This tiny development sample establishes that the pipeline runs and gives us a
baseline to inspect; it does not establish general navigation accuracy.

Reproduce the sample after starting the server:

```powershell
.venv\Scripts\python.exe -m wiki_speedrun evaluate --tasks examples/baseline_tasks.json
.venv\Scripts\python.exe -m wiki_speedrun evaluate --tasks examples/baseline_tasks.json --policy random
```

Local raw outputs from this test, ignored by Git:

- [One-click Kev run](../runs/20261022T013331624118Z-kev/episodes.jsonl)
- [Ten-task Kev traces](../runs/20261022T013409861117Z-kev/episodes.jsonl)
- [Ten-task Kev metrics](../runs/20261022T013409861117Z-kev/metrics.json)
- [Random metrics](../runs/20261022T013403348953Z-random/metrics.json)

All 17 application tests passed, including a CLI run that saves and checks the
result artifacts. CUDA was separately checked with an actual matrix operation.
