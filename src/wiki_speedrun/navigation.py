"""Backend-neutral navigation loop, shared by remote and local evaluation."""
import time
from .environment import NavigationEnvironment
from .evaluation import summarize


def evaluate_policy(policy, graph, tasks, max_steps, *, start_index=0,
                    on_task_start=None, on_transition=None, on_episode=None,
                    on_rejected=None):
    episodes = []
    for index, task in enumerate(tasks, start_index):
        if on_task_start:
            on_task_start(task, index)
        env = NavigationEnvironment(graph, task.start, task.target, max_steps)
        decisions = []
        episode_id = f'eval-{index:06d}'
        failure = None
        while not env.done:
            source, history = env.current, list(env.history)
            choice = policy.choose(env)
            nodes = choice.get('nodes', [])
            decisions.append({'latency_ms': choice.get('latency_ms', 0), 'nodes': len(nodes)})
            if 'rejected' in choice:
                failure = choice['rejected']['failure']
                if on_rejected:
                    on_rejected({'episode_id': episode_id, 'pair': {'start': task.start, 'target': task.target},
                                 'step': env.steps, **choice['rejected']})
                break
            env.step(choice['article'])
            if on_transition:
                on_transition({'schema_version': 1, 'episode_id': episode_id,
                    'pair': {'start': task.start, 'target': task.target}, 'step': env.steps - 1,
                    'source': source, 'target': task.target, 'history': history,
                    'selected_link_id': choice['selected_link_id'], 'selected_article': choice['article'],
                    'next_article': env.current, 'termination_reason': env.termination_reason,
                    'nodes': nodes, 'model_calls': len(nodes), 'latency_ms': choice.get('latency_ms', 0)})
        episode = {'schema_version': 1, 'episode_id': episode_id,
                   'pair': {'start': task.start, 'target': task.target}, 'start': task.start,
                   'target': task.target, 'path': list(env.history), 'steps': env.steps,
                   'success': env.success, 'termination_reason': failure or env.termination_reason,
                   'model_calls': sum(d['nodes'] for d in decisions), 'decisions': decisions}
        episodes.append(episode)
        if on_episode:
            on_episode(episode)
    metrics = summarize(episodes, graph)
    metrics.update({'schema_version': 1, 'failure_counts': {
        reason: sum(e['termination_reason'] == reason for e in episodes)
        for reason in ('step_limit', 'dead_end', 'invalid_link', 'invalid_page', 'invalid_response')},
        'total_model_calls': sum(e['model_calls'] for e in episodes),
        'mean_model_calls': sum(e['model_calls'] for e in episodes) / len(episodes) if episodes else 0,
        'mean_success_steps': (sum(e['steps'] for e in episodes if e['success']) /
                               sum(e['success'] for e in episodes)) if any(e['success'] for e in episodes) else None,
        'episodes': episodes})
    return metrics
