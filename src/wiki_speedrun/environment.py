"""State and transition rules for a Wikipedia navigation episode."""


class NavigationEnvironment:
    def __init__(self, graph, start, target, max_steps=12):
        if start not in graph.articles:
            raise ValueError(f"Unknown start article: {start}")
        if target not in graph.articles:
            raise ValueError(f"Unknown target article: {target}")
        if max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        self.graph = graph
        self.current = start
        self.target = target
        self.max_steps = max_steps
        self.history = [start]
        self.steps = 0

    def legal_moves(self):
        """Return every outgoing link in source order, including duplicates."""
        return list(self.graph.links.get(self.current, []))

    @property
    def success(self):
        return self.current == self.target

    @property
    def termination_reason(self):
        if self.success:
            return "success"
        if self.steps >= self.max_steps:
            return "step_limit"
        if not self.legal_moves():
            return "dead_end"
        return None

    @property
    def done(self):
        return self.termination_reason is not None

    def step(self, article):
        if self.done:
            raise RuntimeError("Episode is already done")
        if article not in self.legal_moves():
            raise ValueError(f"{article!r} is not an outgoing link from {self.current!r}")
        self.current = article
        self.history.append(article)
        self.steps += 1
