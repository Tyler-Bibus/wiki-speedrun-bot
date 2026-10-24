"""Validate saved navigation evidence and compute paired comparisons."""
import itertools
import json
import math
from pathlib import Path
import random

from .data.load import load_graph
from .data.tasks import Task
from .evaluation import summarize
from .remote import read_rows, usage_metrics
from .training.checkpoints import graph_fingerprint, object_fingerprint, sha256_file


PROTOCOLS = {'matched_kev_tournament', 'llm_full_list_json', 'llm_full_list_json_object'}


def validate_tournament(row, graph, config, budgeter):
    from .choices import _fit_state, _groups, make_request
    from .data.load import display_title
    from types import SimpleNamespace
    cfg = SimpleNamespace(**config)
    moves = graph.links[row['source']]
    if not row['nodes']:
        if len(moves) != 1 or row['next_article'] != moves[0]:
            raise ValueError('Unexpected no-inference transition')
        return
    state = {'current_title': display_title(row['source']), 'target_title': display_title(row['target']),
             'current_excerpt': graph.texts.get(row['source'], '')[:cfg.excerpt_chars],
             'history': [display_title(a) for a in row['history']]}
    state, shortening = _fit_state(budgeter, state, cfg)
    candidates = [f'link_{i}' for i in range(len(moves))]
    mapping = dict(zip(candidates, moves))
    nodes = iter(row['nodes'])
    round_number = 0
    while len(candidates) > 1:
        winners = []
        for index, group in enumerate(_groups(budgeter, state, candidates, mapping, cfg)):
            if len(group) == 1:
                winners += group
                continue
            node = next(nodes, None)
            expected = make_request(state, group, [mapping[i] for i in group])
            if (node is None or node['request'] != expected or node['candidate_ids'] != group
                    or node['round'] != round_number or node['node'] != index
                    or node['selected_link_id'] not in group or node['state_shortening'] != shortening):
                raise ValueError('Recorded tournament differs from matched protocol')
            if node['token_counts'] != budgeter.token_counts(expected):
                raise ValueError('Recorded token counts differ')
            winners.append(node['selected_link_id'])
        candidates = winners
        round_number += 1
    if next(nodes, None) is not None or mapping[candidates[0]] != row['next_article']:
        raise ValueError('Tournament trace/path differs')


def validate_run(folder, graph=None):
    folder = Path(folder)
    config = json.loads((folder / 'config.json').read_text())
    provenance = json.loads((folder / 'provenance.json').read_text())
    metrics = json.loads((folder / 'metrics.json').read_text())
    status = json.loads((folder / 'status.json').read_text())
    if status['status'] != 'complete' or metrics['status'] != 'complete':
        raise ValueError(f'{folder}: incomplete runs cannot enter paired results')
    graph = graph or load_graph(Path(config['data_dir']))
    if graph_fingerprint(graph) != provenance['graph_sha256']:
        raise ValueError('Graph/text hash mismatch')
    episodes = read_rows(folder / 'episodes.jsonl')
    pairs = [(e['start'], e['target']) for e in episodes]
    if len(set(pairs)) != len(pairs) or len(pairs) != metrics['intended_tasks']:
        raise ValueError('Missing or duplicate completed pairs')
    if object_fingerprint([{'start': s, 'target': t} for s, t in pairs]) != provenance['ordered_tasks_sha256']:
        raise ValueError('Ordered task coverage mismatch')
    requests = read_rows(folder / 'requests.jsonl')
    responses = {}
    for record in requests:
        if 'result' in record and 'error' not in record:
            responses[(record.get('session'), record.get('episode_id'), record.get('step'), record.get('round'), record.get('node'))] = record
    transitions = read_rows(folder / 'transitions.jsonl')
    by_episode = {}
    for row in transitions:
        by_episode.setdefault((row['episode_id'], row.get('session')), []).append(row)
    budgeter = None
    if provenance['protocol'] == 'matched_kev_tournament':
        from .models.kev_budget import KevContextBudgeter
        from types import SimpleNamespace
        budgeter = KevContextBudgeter(config['tokenizer'], SimpleNamespace(**config))
    calls = 0
    for episode in episodes:
        path = episode['path']
        if not path or path[0] != episode['start'] or len(path) != episode['steps'] + 1:
            raise ValueError('Invalid path length/start')
        if episode['success'] != (path[-1] == episode['target']):
            raise ValueError('Success inconsistent with final page')
        if episode['steps'] > config['max_steps']:
            raise ValueError('Click ceiling exceeded')
        for source, target in zip(path, path[1:]):
            if target not in graph.links.get(source, []):
                raise ValueError('Illegal graph edge in path')
        rows = by_episode.get((episode['episode_id'], episode.get('session')), [])
        if len(rows) != episode['steps']:
            raise ValueError('Transition/path count mismatch')
        for i, row in enumerate(rows):
            if row['step'] != i or row['source'] != path[i] or row['next_article'] != path[i+1] or row['history'] != path[:i+1]:
                raise ValueError('Transition/path content mismatch')
            if budgeter is not None:
                validate_tournament(row, graph, config, budgeter)
                if requests:
                    for node in row['nodes']:
                        record = responses.get((episode.get('session'), episode['episode_id'], i, node['round'], node['node']))
                        expected = dict(node['request'], model=config['model'])
                        if record is None or record['request'] != expected or record['result']['choice'] != node['selected_link_id']:
                            raise ValueError('Response/tournament decision mismatch')
            for node in row.get('nodes', []):
                if 'candidate_ids' in node and node['selected_link_id'] not in node['candidate_ids']:
                    raise ValueError('Tournament selected ID outside its group')
        calls += episode['model_calls']
    rejected = read_rows(folder / 'rejected-decisions.jsonl')
    for e in episodes:
        if e['termination_reason'].startswith('invalid_'):
            rows = [r for r in rejected if r['episode_id'] == e['episode_id'] and r.get('session') == e.get('session')]
            if len(rows) != 1 or rows[0]['source'] != e['path'][-1] or rows[0]['step'] != e['steps'] or rows[0]['failure'] != e['termination_reason']:
                raise ValueError('Rejected decision/episode mismatch')
    recomputed = summarize(episodes, graph)
    for key in ('episode_count', 'success_count', 'success_rate', 'mean_steps', 'total_revisit_count', 'mean_shortest_path_overhead'):
        if metrics.get(key) != recomputed[key]:
            raise ValueError('Saved aggregate mismatch: ' + key)
    if (folder / 'requests.jsonl').exists():
        accounting = usage_metrics(read_rows(folder / 'requests.jsonl'))
        for key in ('known_cost', 'http_attempts', 'tokens', 'unknown_cost_attempts'):
            if accounting[key] != metrics[key]:
                raise ValueError('Saved usage total mismatch: ' + key)
    return {'folder': str(folder), 'config': config, 'provenance': provenance,
            'metrics': metrics, 'episodes': episodes, 'model_calls': calls}


def paired(a, b):
    aa = {(e['start'], e['target']): e for e in a['episodes']}
    bb = {(e['start'], e['target']): e for e in b['episodes']}
    if set(aa) != set(bb):
        raise ValueError('Paired comparison requires identical task sets')
    differences = [int(aa[k]['success']) - int(bb[k]['success']) for k in aa]
    wins = sum(x == 1 for x in differences)
    losses = sum(x == -1 for x in differences)
    common = [k for k in aa if aa[k]['success'] and bb[k]['success']]
    rng = random.Random(42)
    boot = sorted(sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(2000))
    discordant = wins + losses
    p = min(1.0, 2 * sum(math.comb(discordant, k) for k in range(min(wins, losses) + 1)) / 2**discordant) if discordant else 1.0
    return {'a_only_success': wins, 'b_only_success': losses, 'both_success': len(common),
            'success_rate_difference': sum(differences) / len(differences),
            'paired_bootstrap_95_ci': [boot[49], boot[1949]], 'mcnemar_exact_p': p,
            'common_success_mean_click_difference': sum(aa[k]['steps'] - bb[k]['steps'] for k in common) / len(common) if common else None}


def write_comparison(folders, output, *, allow_system_protocols=False, graph=None):
    runs = [validate_run(f, graph) for f in folders]
    if not runs:
        raise ValueError('No complete runs to compare')
    first = runs[0]['provenance']
    for run in runs:
        p = run['provenance']
        if any(run['config'][k] != runs[0]['config'][k] for k in ('max_steps', 'excerpt_chars')):
            raise ValueError('Navigation click/excerpt budget mismatch')
        if p['protocol'] not in PROTOCOLS:
            raise ValueError('Unsupported protocol')
        if any(p[k] != first[k] for k in ('graph_sha256', 'ordered_tasks_sha256', 'selection')):
            raise ValueError('Comparison data/task/selection hash mismatch')
        if not allow_system_protocols and p['protocol'] != first['protocol']:
            raise ValueError('Protocol mismatch: opt in to system comparison explicitly')
    matched = [r for r in runs if r['provenance']['protocol'] == 'matched_kev_tournament']
    if matched:
        reference = matched[0]
        for run in matched[1:]:
            for key in ('max_steps', 'excerpt_chars', 'max_state_tokens', 'max_row_tokens', 'node_max_choices'):
                if run['config'][key] != reference['config'][key]:
                    raise ValueError('Matched tournament configuration mismatch')
            if run['provenance']['tokenizer']['tokenizer_sha256'] != reference['provenance']['tokenizer']['tokenizer_sha256']:
                raise ValueError('Matched tokenizer mismatch')
            for key in ('encoding_sha256', 'api_sha256', 'kev_revision'):
                if run['provenance']['tokenizer'][key] != reference['provenance']['tokenizer'][key]:
                    raise ValueError('Matched encoding mismatch')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# Navigation comparison', '',
             'Results are recomputed from complete saved episodes after checking graph edges, task coverage, transitions, and usage totals.', '',
             'LLMs see the full list and history; decision models use matched Kev tournaments. These are system comparisons when protocols differ.', '',
             '| Model | Protocol | Success | Mean successful clicks | Revisits | Shortest-path excess | Model calls | HTTP attempts | Known cost / unknown attempts |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in runs:
        m, p = r['metrics'], r['provenance']
        lines.append(f"| {p['requested_model']} | {p['protocol']} | {m['success_count']}/{m['episode_count']} ({m['success_rate']:.1%}) | {m['mean_success_steps']} | {m['total_revisit_count']} | {m['mean_shortest_path_overhead']} | {r['model_calls']} | {m.get('http_attempts', 'local')} | {m.get('known_cost', 'local GPU')} / {m.get('unknown_cost_attempts', 'n/a')} |")
    results = []
    for a, b in itertools.combinations(runs, 2):
        result = paired(a, b)
        result.update(a=a['provenance']['requested_model'], b=b['provenance']['requested_model'])
        results.append(result)
    lines += ['', 'Paired wins/losses, common-solved click differences, seeded 2,000-resample paired bootstrap intervals, and exact McNemar p-values are saved in the adjacent JSON.', '',
              '## Invalid decisions, usage, and latency', '',
              '| Model | Invalid link / page / response | Attempted decisions | Legal clicks | Input / output tokens | API latency ms | Wall seconds | Known cost per solved task |',
              '| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |',
              'All costs include failures, retries, and abandoned episode attempts. Missing cost is unknown. Network-inclusive API latency and local GPU latency have different execution conditions.', '',
              'The historical SFT test outcomes had already been viewed before this comparison. The fixed SFT and RL checkpoints were selected using development outcomes.', '', '## Run artifacts', '']
    details = []
    for r in runs:
        m = r['metrics']; f = m.get('failure_counts', {}); t = m.get('tokens', {})
        details.append(f"| {r['provenance']['requested_model']} | {f.get('invalid_link', 0)} / {f.get('invalid_page', 0)} / {f.get('invalid_response', 0)} | {m.get('submitted_model_calls', r['model_calls'])} | {sum(e['steps'] for e in r['episodes'])} | {t.get('input_tokens') or t.get('prompt_tokens')} / {t.get('output_tokens') or t.get('completion_tokens')} | {m.get('remote_latency_ms', m.get('total_inference_latency_ms'))} | {m.get('wall_seconds', 'unrecorded')} | {m.get('known_cost_per_solved_task', 'local GPU')} |")
    insert_at = lines.index('All costs include failures, retries, and abandoned episode attempts. Missing cost is unknown. Network-inclusive API latency and local GPU latency have different execution conditions.')
    lines[insert_at:insert_at] = details + ['']
    lines += [f"- `{r['folder']}`" for r in runs]
    output.write_text('\n'.join(lines) + '\n')
    output.with_suffix('.json').write_text(json.dumps({'runs': [{k: v for k, v in r.items() if k != 'episodes'} for r in runs],
                                                     'paired': results}, indent=2, allow_nan=False))
    return runs
