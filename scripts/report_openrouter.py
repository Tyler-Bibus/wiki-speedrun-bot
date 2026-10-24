#!/usr/bin/env python3
"""Regenerate the saved full-split report offline, without spending API credit."""
from collections import Counter
from datetime import datetime
import fcntl
import json
import math
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

from wiki_speedrun.comparison import write_comparison
from wiki_speedrun.data.load import load_graph
from wiki_speedrun.evaluation import summarize
from wiki_speedrun.remote import read_rows, selected_tasks, usage_metrics
from wiki_speedrun.remote_config import load_remote_config
from wiki_speedrun.training.checkpoints import graph_fingerprint, sha256_file, write_json_atomic

ROOT = Path('runs/openrouter-full')
NAMES = {'kev': 'Released Kev 0.8B', 'sft': 'SFT (update 1700)', 'rl': 'SFT+RL (update 75)',
         'minimax': 'MiniMax M3.1 Flash', 'deepseek': 'DeepSeek V4.1 Flash 1017',
         'mammoth': 'Mammoth 1', 'elephant': 'Elephant 1'}
ORDER = list(NAMES)
REASONS = ('step_limit', 'dead_end', 'invalid_link', 'invalid_page', 'invalid_response')


def load(path):
    return json.loads(Path(path).read_text())


def lock_held(name):
    with (ROOT / name).open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return False
        except BlockingIOError:
            return True


def validate_prefix(folder, split, tasks, selection, graph, graph_hash):
    """Check saved coverage/paths/accounting without promoting an incomplete run."""
    status = load(folder / 'status.json'); metrics = load(folder / 'metrics.json')
    provenance = load(folder / 'provenance.json')
    episodes = read_rows(folder / 'episodes.jsonl')
    actual = [(e['start'], e['target']) for e in episodes]
    expected = [(t.start, t.target) for t in tasks[:len(episodes)]]
    assert actual == expected and len(actual) == len(set(actual)), 'Invalid completed prefix'
    assert len(episodes) <= len(tasks)
    assert status['completed_tasks'] == len(episodes) == metrics['episode_count']
    assert status['intended_tasks'] == len(tasks) == metrics['intended_tasks']
    assert provenance['selection'] == selection and provenance['graph_sha256'] == graph_hash
    for episode in episodes:
        path = episode['path']
        assert path[0] == episode['start'] and len(path) == episode['steps'] + 1
        assert episode['success'] == (path[-1] == episode['target'])
        assert all(b in graph.links.get(a, []) for a, b in zip(path, path[1:]))
    computed = summarize(episodes, graph)
    for key in ('episode_count', 'success_count', 'success_rate', 'mean_steps', 'total_revisit_count'):
        assert metrics[key] == computed[key], 'Prefix aggregate mismatch: ' + key
    accounting = usage_metrics(read_rows(folder / 'requests.jsonl'))
    for key in ('http_attempts', 'known_cost', 'unknown_cost_attempts', 'tokens'):
        assert accounting[key] == metrics[key], 'Prefix usage mismatch: ' + key
    return {'folder': str(folder), 'status': status, 'metrics': metrics, 'episodes': episodes}


def percentage(value):
    return f'{value * 100:.2f}%'


def success_cell(run):
    m = run['metrics']
    return f"{m['success_count']}/{m['episode_count']} ({percentage(m['success_rate'])})"


def number(value, places=2):
    return 'unavailable' if value is None else f'{value:.{places}f}'


def main():
    state = load(ROOT / 'benchmark.json'); job = load(ROOT / 'job.json')
    budget = load(ROOT / 'budget.json'); recovery = load(ROOT / 'recovery.json')
    graph = load_graph(Path('data')); graph_hash = graph_fingerprint(graph)
    config = load_remote_config('configs/jev.toml')
    task_info = {split: selected_tasks(config, split) for split in ('dev', 'test')}
    complete, prefixes = {}, {}
    for split in ('dev', 'test'):
        folders, labels = [], []
        for name in ORDER:
            label = name + '-' + split
            if label not in state['runs']:
                continue
            folder = Path(state['runs'][label]); status = load(folder / 'status.json')
            if status['status'] == 'complete':
                assert load(folder / 'metrics.json')['episode_count'] == len(task_info[split][0])
                assert load(folder / 'provenance.json')['selection'] == task_info[split][1]
                folders.append(folder); labels.append(label)
            else:
                prefixes[label] = validate_prefix(folder, split, *task_info[split], graph, graph_hash)
        runs = write_comparison(folders, ROOT / (split + '-results.md'), allow_system_protocols=True, graph=graph)
        complete.update(dict(zip(labels, runs)))
        print(f'Validated {split}: {len(runs)} complete full-split runs', flush=True)

    all_receipts = [row for p in ROOT.rglob('requests.jsonl') for row in read_rows(p)]
    accounting = usage_metrics(all_receipts)
    assert math.isclose(accounting['known_cost'], budget['known_cost'], abs_tol=1e-9)
    assert accounting['http_attempts'] == budget['http_attempts']
    assert accounting['unknown_cost_attempts'] == budget['unknown_cost_attempts']
    workers = {'batch': lock_held('worker.lock'), 'recovery': lock_held('recovery.lock')}
    protected = Path('/tmp/openrouter-protected.json')
    protected_count = None
    if protected.exists():
        before = load(protected)
        assert all(sha256_file(Path(p)) == value for p, value in before.items())
        protected_count = len(before)

    stamp = datetime.now(ZoneInfo('America/Chicago')).strftime('%B %d, %Y, %I:%M %p %Z')
    remote_completed = sum(len(r['episodes']) for label, r in complete.items() if label.split('-')[0] not in ('kev', 'sft', 'rl'))
    remote_completed += sum(len(r['episodes']) for r in prefixes.values())
    status_text = ('The experiment is paused (`needs_attention`); neither the batch worker nor its recovery worker is active.'
                   if not any(workers.values()) else 'The experiment has an active worker; this report is a saved-results snapshot.')
    overview = ['# Validation and test evaluation results', '', f'Updated {stamp}.', '', status_text, '',
                '**The complete seven-system comparison is not finished.** MiniMax has complete validation and test runs; '
                'DeepSeek has a complete validation run. All three local checkpoints have complete runs on both splits. '
                'DeepSeek test, Mammoth, and Elephant remain incomplete because of provider errors and daily quotas. '
                'Only completed full splits enter the comparison tables below.', '',
                '## Completed full-split results', '',
                '| System | Validation: 179 pairs | Test: 175 pairs |', '| --- | ---: | ---: |']
    for name in ORDER:
        overview.append('| ' + NAMES[name] + ' | ' + ' | '.join(
            success_cell(complete[name + '-' + split]) if name + '-' + split in complete else 'Not complete'
            for split in ('dev', 'test')) + ' |')
    overview += ['', 'MiniMax solved one more test pair than SFT, a difference of 0.57 percentage points. '
                 'That count alone does not establish a reliable improvement. DeepSeek currently has no full-test result. '
                 'The original five-task pilot is excluded from every table in this report.', '']
    lines = overview + ['## Coverage and stopped work', '',
             f'{remote_completed}/1,416 intended remote episodes are committed. Local coverage is complete: '
             '179 validation and 175 test episodes for each of three checkpoints.', '',
             '| Unfinished run | Episodes completed / intended | Successes in the completed prefix | Stop reason |',
             '| --- | ---: | ---: | --- |']
    for name in ('deepseek', 'mammoth', 'elephant'):
        for split in ('dev', 'test'):
            label = name + '-' + split
            if label in complete:
                continue
            entry = prefixes.get(label)
            error = state.get('errors', {}).get(label, {})
            count = len(entry['episodes']) if entry else 0
            successes = entry['metrics']['success_count'] if entry else 0
            reason = error.get('kind', 'not started')
            lines.append(f"| {NAMES[name]} {'validation' if split == 'dev' else 'test'} | {count}/{len(task_info[split][0])} | {successes} | {reason} |")
    lines += ['', 'The prefix counts above are progress records, not full-split success rates. No missing task is '
              'silently skipped or reclassified as a navigation failure. The interrupted episode can restart; '
              'committed episodes, including invalid LLM decisions, remain fixed.', '',
              'DeepSeek returned embedded provider 502 completion errors. The normal worker treated these as '
              'API-envelope failures and stopped; a separate recovery worker resumed the frozen experiment three times. '
              'It completed DeepSeek validation, then exhausted its recovery passes during test. '
              'Mammoth and Elephant received HTTP 429 daily-quota errors. '
              'Neither the batch nor recovery process is currently waiting to retry automatically.', '',
              '## Paired comparisons and uncertainty', '',
              'Rows compare the left system against the right on exactly the same complete split. '
              'Success differences and 95% intervals are percentage points. Click differences use only pairs both systems solved; '
              'positive values mean more clicks for the left system. Intervals use 2,000 paired bootstrap resamples with seed 42; '
              'p-values are exact two-sided McNemar tests. These exploratory comparisons are not corrected for multiple testing.', '',
              '| Split | Left vs right | Left-only / right-only successes | Both solved | Success difference (95% interval), pp | Common-solved click difference | McNemar p |',
              '| --- | --- | ---: | ---: | --- | ---: | ---: |']
    selected_pairs = [('minimax', 'sft'), ('minimax', 'rl'), ('rl', 'sft'), ('deepseek', 'sft'), ('minimax', 'deepseek')]
    paired_by_split = {}
    for split in ('dev', 'test'):
        data = load(ROOT / (split + '-results.json')); paired_by_split[split] = data['paired']
        for left, right in selected_pairs:
            a, b = complete.get(left + '-' + split), complete.get(right + '-' + split)
            if not a or not b:
                continue
            aid, bid = a['provenance']['requested_model'], b['provenance']['requested_model']
            pair = next((p for p in data['paired'] if {p['a'], p['b']} == {aid, bid}), None)
            reverse = pair['a'] != aid
            sign = -1 if reverse else 1
            wins = pair['b_only_success'] if reverse else pair['a_only_success']
            losses = pair['a_only_success'] if reverse else pair['b_only_success']
            lo, hi = pair['paired_bootstrap_95_ci']
            if reverse: lo, hi = -hi, -lo
            clicks = pair['common_success_mean_click_difference']
            if clicks is not None: clicks *= sign
            lines.append(f"| {'Validation' if split == 'dev' else 'Test'} | {NAMES[left]} vs {NAMES[right]} | "
                         f"{wins} / {losses} | {pair['both_success']} | {sign * pair['success_rate_difference'] * 100:+.2f} "
                         f"[{lo * 100:+.2f}, {hi * 100:+.2f}] | {number(clicks, 3)} | {pair['mcnemar_exact_p']:.4f} |")
    lines += ['', 'All pairwise statistics, including comparisons with released Kev, are saved in '
              '`runs/openrouter-full/dev-results.json` and `test-results.json`. '
              'Different action-presentation protocols are labeled system comparisons; these results do not isolate model weights.', '',
              '## Navigation efficiency on completed splits', '',
              '| System / split | Successful-route mean clicks | Mean shortest-path excess | Total revisits | Model calls in committed episodes |',
              '| --- | ---: | ---: | ---: | ---: |']
    for label, run in complete.items():
        name, split = label.rsplit('-', 1); m = run['metrics']
        lines.append(f"| {NAMES[name]} / {split} | {number(m['mean_success_steps'])} | "
                     f"{number(m['mean_shortest_path_overhead'])} | {m['total_revisit_count']} | {run['model_calls']} |")
    lines += ['', 'Successful-route means alone compare different solved subsets. Use the paired common-solved '
              'click differences above for route-efficiency comparisons between systems.', '',
              '## LLM failures on completed full splits', '',
              '| System / split | Step limit | Dead end | Invalid link | Invalid page | Invalid response | Invalid episodes / all episodes | Invalid decisions / committed decisions |',
              '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label, run in complete.items():
        name, split = label.rsplit('-', 1)
        if name not in ('minimax', 'deepseek'): continue
        reasons = Counter(e['termination_reason'] for e in run['episodes'])
        invalid = sum(reasons[k] for k in ('invalid_link', 'invalid_page', 'invalid_response'))
        lines.append(f"| {NAMES[name]} / {split} | " + ' | '.join(str(reasons[k]) for k in REASONS) +
                     f" | {invalid}/{len(run['episodes'])} | {invalid}/{run['model_calls']} |")
    lines += ['', 'An invalid decision ends its episode immediately, stays in the split denominator, and adds no illegal click. '
              'The attempted-decision denominator in this table covers committed episodes; infrastructure errors '
              'and abandoned episode attempts remain in the separate HTTP/cost accounting.', '',
              '## API usage and execution time', '',
              f"Known suite cost: **${accounting['known_cost']:.6f}** of the **${budget['max_total_cost']:.2f}** shared ceiling. "
              f"HTTP attempts: **{accounting['http_attempts']:,}**; attempts with unknown cost: **{accounting['unknown_cost_attempts']}**. "
              'The stop was caused by service errors/quotas, not the spending ceiling. This suite total includes probes, '
              'failed attempts, and abandoned episode attempts, but excludes the earlier separate pilot.', '',
              '| Run | Coverage | HTTP attempts | Known cost | Unknown-cost attempts | Input tokens | Output tokens | Reported reasoning tokens | API seconds / attempt | Active wall minutes |',
              '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    remote_entries = [(label, run['metrics'], 'complete') for label, run in complete.items() if label.split('-')[0] in ('minimax', 'deepseek', 'mammoth', 'elephant')]
    remote_entries += [(label, run['metrics'], 'incomplete') for label, run in prefixes.items()]
    for label, m, coverage in remote_entries:
        tokens = m['tokens']; inp = tokens.get('input_tokens'); out = tokens.get('output_tokens')
        if inp is None: inp = tokens.get('prompt_tokens')
        if out is None: out = tokens.get('completion_tokens')
        seconds = m['remote_latency_ms'] / m['http_attempts'] / 1000 if m['http_attempts'] else None
        lines.append(f"| {label} | {coverage} | {m['http_attempts']} | ${m['known_cost']:.6f} | "
                     f"{m['unknown_cost_attempts']} | {inp} | {out} | {tokens.get('reasoning_tokens')} | "
                     f"{number(seconds)} | {number(m['wall_seconds'] / 60)} |")
    lines += ['', 'These are provider-reported token/cost totals, including failed and abandoned work within each run. '
              'DeepSeek test has inconsistent provider usage metadata: its reported reasoning-token total exceeds its '
              'reported completion-token total. Those figures are retained as reported; cost uses the receipt `usage.cost`, '
              'not a calculation from token counts. Probe and setup errors account for remaining suite attempts outside the run rows.', '',
              'API latency includes provider/network time. Active wall time includes tournament budgeting, throttling, '
              'retries and persistence during execution, and excludes model setup and time between stopped worker passes. '
              'Mammoth also accumulated roughly ' + number(prefixes['mammoth-dev']['metrics']['throttle_wait_ms'] / 60000) +
              ' minutes of configured pacing. Local RTX 3090 latency has different execution conditions.', '',
              '## Protocol and provenance', '',
              'All systems use the frozen graph, the same ordered task pairs, a 12-click ceiling and 1,500-character '
              'initial excerpts. Decision systems use `matched_kev_tournament`: exact pinned Kev encoding, 64-choice nodes, '
              '1,024 state tokens and 4,096 row tokens. LLMs receive the entire outgoing list and full history in one call. '
              'DeepSeek uses `llm_full_list_json` with strict JSON schema. MiniMax uses the user-approved '
              '`llm_full_list_json_object` with strict local validation. Both LLMs request temperature 0, reasoning effort '
              '`low` and at most 2,048 completion tokens. No fallback model or prompt correction is used.', '',
              'The fixed local checkpoints are released Kev '
              '`checkpoints/comparison-untuned-kev-0.8b-bf75a6a`, SFT '
              '`checkpoints/20261022T025733.517853Z-sft/update-001700`, and selected RL '
              '`checkpoints/20261022T132731.513046Z-rl/update-000075`. '
              'Historical SFT test outcomes had already been viewed. Test outcomes did not select checkpoints or prompts.', '',
              f'Graph/text SHA-256: `{graph_hash}`.', '']
    for split in ('dev', 'test'):
        selection = task_info[split][1]
        lines.append(f"{split} task-file SHA-256: `{selection['tasks_file_sha256']}`; manifest SHA-256: `{selection['manifest_sha256']}`.")
    lines += ['', 'Completed runs were validated for exact task coverage, legal edges, transition/path consistency, '
              'reconstructed matched tournaments and token counts, rejected-decision records and recomputed aggregate/usage totals. '
              'Incomplete prefixes were checked separately for ordered coverage, legal paths and matching counts/usage; '
              'they were not promoted to complete runs. '
              + (f'All {protected_count} protected checkpoint files and pointers still match their pre-evaluation fingerprints.' if protected_count else ''), '',
              '## Reproduction and artifact locations', '',
              'Rebuild this report from existing traces, offline and without new API calls:', '',
              '```bash', '.venv/bin/python scripts/report_openrouter.py', '```', '',
              'Resume the remaining benchmark work with the frozen configurations and existing $9 shared limit:', '',
              '```bash', '.venv/bin/python -m wiki_speedrun.batch --background', '```', '',
              'Resume preserves committed episodes and restarts an interrupted episode. Embedded provider errors may '
              'stop the current worker again; daily quota availability is external. A successful report rebuild does '
              'not mean the unfinished evaluations completed.', '',
              '- Full validated comparison artifacts: `runs/openrouter-full/dev-results.md` and `test-results.md`, with adjacent JSON.',
              '- Execution state: `runs/openrouter-full/job.json`, `benchmark.json`, `budget.json`, `recovery.json`.',
              '- Error/recovery history: `runs/openrouter-full/worker.log` and `recovery.log`.',
              '- Original implementation/pilot report: [jev-pilot-results.md](jev-pilot-results.md).', '',
              '### Source run directories', '']
    lines += [f'- `{label}`: `{folder}`.' for label, folder in sorted(state['runs'].items())]
    lines += ['']
    docs = Path('docs'); pilot = docs / 'jev-pilot-results.md'
    if not pilot.exists():
        shutil.copyfile(docs / 'jev-results.md', pilot)
    (docs / 'jev-full-results.md').write_text('\n'.join(lines))
    landing = overview + ['## Remaining work and cost', '',
        'The batch and recovery worker have stopped after provider errors and exhausted daily quotas. '
        'DeepSeek test has 45/175 episodes committed; Mammoth validation has 99/179. Mammoth test and '
        'both Elephant splits have no committed episodes. These incomplete runs have no full-split success rates.', '',
        f"Known suite spend is ${accounting['known_cost']:.6f}; {accounting['unknown_cost_attempts']} attempts lack cost receipts. "
        'The configured shared limit remains $9.', '',
        'Read the [full report](jev-full-results.md) for paired uncertainty estimates, route efficiency, '
        'invalid-output counts, costs, timing, provenance and exact remaining coverage. '
        'The [original implementation and pilot report](jev-pilot-results.md) is preserved separately.', '',
        'Rebuild the current full-split report without API calls:', '', '```bash',
        '.venv/bin/python scripts/report_openrouter.py', '```', '']
    (docs / 'jev-results.md').write_text('\n'.join(landing))
    interim = docs / 'jev-full-interim-results.md'
    previous = interim.read_text()
    note = '**Superseded:** see [the updated full-split report](jev-full-results.md), which also includes completed DeepSeek validation and current stop conditions.\n\n'
    if not previous.startswith('**Superseded:**'):
        interim.write_text(note + previous)
    write_json_atomic(ROOT / 'report-snapshot.json', {
        'updated_at': stamp, 'job_status': job['status'], 'workers_active': workers,
        'complete_runs': list(complete), 'incomplete_runs': list(prefixes),
        'remote_episodes_committed': remote_completed, 'remote_episodes_intended': 1416,
        'accounting': accounting, 'recovery_passes': recovery['recovery_passes'],
        'report_generator_sha256': sha256_file(Path(__file__)),
    })
    print('Updated docs/jev-full-results.md and docs/jev-results.md', flush=True)


if __name__ == '__main__':
    main()
