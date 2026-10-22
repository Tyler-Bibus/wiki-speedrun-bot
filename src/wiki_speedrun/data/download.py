"""Download and prepare Stanford SNAP's Wikispeedia dataset."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import tarfile
import urllib.request
from pathlib import Path, PurePosixPath

from .load import load_raw_graph

GRAPH_URL = "https://snap.stanford.edu/data/wikispeedia/wikispeedia_paths-and-graph.tar.gz"
TEXT_URL = "https://snap.stanford.edu/data/wikispeedia/wikispeedia_articles_plaintext.tar.gz"


def _safe_extract(archive: Path, destination: Path) -> None:
    """Extract regular files/directories only, rejecting paths outside destination."""
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar.getmembers():
            name = member.name.replace("\\", "/")
            posix = PurePosixPath(name)
            if posix.is_absolute() or ".." in posix.parts or member.issym() or member.islnk():
                continue
            target = (destination / Path(*posix.parts)).resolve()
            if target != root and root not in target.parents:
                continue
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is not None:
                    with source, target.open("wb") as output:
                        shutil.copyfileobj(source, output)


def _download(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=target.name + ".", suffix=".part", dir=target.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with urllib.request.urlopen(url, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_data(data_dir: Path, include_text: bool = True) -> Path:
    """Download SNAP archives, extract safely, and write processed/graph.json."""
    data_dir = Path(data_dir)
    raw_dir = data_dir / "raw"
    archives = data_dir / "archives"
    graph_archive = archives / "wikispeedia_paths-and-graph.tar.gz"
    if not graph_archive.exists():
        _download(GRAPH_URL, graph_archive)
    _safe_extract(graph_archive, raw_dir)
    if include_text:
        text_archive = archives / "wikispeedia_articles_plaintext.tar.gz"
        if not text_archive.exists():
            _download(TEXT_URL, text_archive)
        _safe_extract(text_archive, raw_dir)

    graph = load_raw_graph(raw_dir)
    processed = data_dir / "processed" / "graph.json"
    processed.parent.mkdir(parents=True, exist_ok=True)
    processed.write_text(json.dumps({
        "articles": sorted(graph.articles),
        "links": graph.links,
        "texts": graph.texts,
    }, ensure_ascii=False), encoding="utf-8")
    return processed
