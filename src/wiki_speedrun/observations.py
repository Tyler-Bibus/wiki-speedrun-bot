"""Build the constrained request sent to the decision model."""


class OptionLimitError(ValueError):
    """Raised when an observation has more links than the request can encode."""


def _display_title(title):
    return title.replace("_", " ")


def build_request(env, excerpt_chars=1500, max_options=255):
    if excerpt_chars < 0:
        raise ValueError("excerpt_chars must be non-negative")
    if max_options < 0:
        raise ValueError("max_options must be non-negative")
    moves = env.legal_moves()
    if len(moves) > max_options:
        raise OptionLimitError(f"{len(moves)} outgoing links exceed max_options={max_options}")
    criteria = {f"link_{i}": _display_title(title) for i, title in enumerate(moves)}
    excerpt = env.graph.texts.get(env.current, "")[:excerpt_chars]
    state = {
        "current_title": _display_title(env.current),
        "target_title": _display_title(env.target),
        "current_excerpt": excerpt,
        "history": [_display_title(title) for title in env.history],
    }
    return {
        "model": "kev-latest",
        "state": state,
        "questions": {
            "next_link": {
                "type": "choice",
                "instructions": (
                    "Reach the target using as few clicks as possible. Use the page excerpt and "
                    "history to avoid loops. Choose only an outgoing link and return its link ID."
                ),
                "criteria": criteria,
            }
        },
    }
