"""Load the prepared Wikispeedia graph and SNAP's extracted text files."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote


@dataclass
class WikiGraph:
    articles: set[str]
    links: dict[str, list[str]]
    texts: dict[str, str] = field(default_factory=dict)


def decode_id(value: str) -> str:
    """Decode SNAP's percent-escaped article IDs while retaining underscores."""
    return unquote(value.strip())


def display_title(article_id: str) -> str:
    return article_id.replace("_", " ")


def _rows(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield line.split("\t")


def _find(root: Path, filename: str) -> Path | None:
    matches = sorted(root.rglob(filename)) if root.exists() else []
    return matches[0] if matches else None


def load_raw_graph(raw_dir: Path) -> WikiGraph:
    """Load articles.tsv and links.tsv from SNAP's extracted archive tree."""
    raw_dir = Path(raw_dir)
    articles_file = _find(raw_dir, "articles.tsv")
    links_file = _find(raw_dir, "links.tsv")
    if articles_file is None or links_file is None:
        raise FileNotFoundError(f"Could not find articles.tsv and links.tsv beneath {raw_dir}")

    articles: set[str] = set()
    for fields in _rows(articles_file):
        if fields:
            articles.add(decode_id(fields[0]))

    links: dict[str, list[str]] = {article: [] for article in articles}
    for fields in _rows(links_file):
        if len(fields) >= 2:
            source, target = decode_id(fields[0]), decode_id(fields[1])
            articles.update((source, target))
            links.setdefault(source, []).append(target)
            links.setdefault(target, [])

    texts: dict[str, str] = {}
    for text_file in sorted(raw_dir.rglob("*.txt")):
        # SNAP plaintext articles use one file per encoded page ID. Skip metadata.
        if text_file.name in {"articles.tsv", "links.tsv"}:
            continue
        article_id = decode_id(text_file.stem)
        if article_id in articles:
            try:
                texts[article_id] = text_file.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
    return WikiGraph(articles=articles, links=links, texts=texts)


def load_graph(data_dir: Path) -> WikiGraph:
    """Load processed/graph.json, or parse raw extracted SNAP files."""
    data_dir = Path(data_dir)
    processed = data_dir / "processed" / "graph.json"
    if processed.is_file():
        payload = json.loads(processed.read_text(encoding="utf-8"))
        articles = {decode_id(str(item)) for item in payload.get("articles", [])}
        links = {
            decode_id(str(source)): [decode_id(str(target)) for target in targets]
            for source, targets in payload.get("links", {}).items()
        }
        texts = {decode_id(str(key)): str(value) for key, value in payload.get("texts", {}).items()}
        for article in articles:
            links.setdefault(article, [])
        return WikiGraph(articles, links, texts)
    return load_raw_graph(data_dir / "raw")


def load_paths(path: Path) -> list[list[str]]:
    """Load completed SNAP paths as their final active page stacks.

    In SNAP's finished-path TSV, the path is column 4 (zero-based 3). A ``<``
    action pops the current page from the active stack. The returned routes are
    therefore loop-erased final stacks, not chronological action traces; use
    the raw TSV path column when training on individual back-button actions.
    """
    result: list[list[str]] = []
    for fields in _rows(Path(path)):
        # SNAP: hashed IP, timestamp, duration, path, rating.
        if len(fields) >= 4:
            encoded = fields[3]
        elif len(fields) == 1:
            # Also accept a file containing just semicolon-separated paths.
            encoded = fields[0]
        else:
            continue
        route: list[str] = []
        for token in encoded.split(";"):
            if token == "<":
                if route:
                    route.pop()
            elif token:
                route.append(decode_id(token))
        if route:
            result.append(route)
    return result
