"""Validate human routes and create frozen pair splits and SFT node examples."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter, deque
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..choices import INSTRUCTIONS, build_supervised_nodes, make_request
from .load import WikiGraph, decode_id, display_title, load_graph


PREPROCESSING_VERSION = 1
BASELINE_TASKS = Path("examples/baseline_tasks.json")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_pair(start: str, target: str) -> tuple[str, str]:
    return start, target


def _pair_split(start: str, target: str, seed: int) -> str:
    first, second = _canonical_pair(start, target)
    value = int.from_bytes(hashlib.sha256(
        f"{seed}\0{first}\0{second}".encode("utf-8")).digest()[:8], "big") % 100
    if value < 80:
        return "train"
    if value < 90:
        return "dev"
    return "test"


def _read_raw_routes(path: Path) -> list[tuple[int, list[str]]]:
    """Read load_paths-compatible final stacks while retaining physical row IDs."""
    routes: list[tuple[int, list[str]]] = []
    from urllib.parse import unquote

    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for row_id, line in enumerate(stream, start=1):
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            fields = text.split("\t")
            if len(fields) >= 4:
                encoded = fields[3]
            elif len(fields) == 1:
                encoded = fields[0]
            else:
                continue
            route: list[str] = []
            for token in encoded.split(";"):
                if token == "<":
                    if route:
                        route.pop()
                elif token:
                    route.append(unquote(token.strip()))
            if route:
                routes.append((row_id, route))
    return routes


def _relative_path(config: Any, name: str) -> Path:
    return Path(getattr(config.data, name))


def _raw_paths_file(data_root: Path) -> Path:
    matches = sorted((data_root / "raw").rglob("paths_finished.tsv"))
    if not matches:
        raise FileNotFoundError(f"Could not find paths_finished.tsv beneath {data_root / 'raw'}")
    return matches[0]


def _graph_fingerprint(data_root: Path, graph: WikiGraph | None = None) -> str:
    graph_path = data_root / "processed" / "graph.json"
    if graph_path.is_file():
        return _sha256_file(graph_path)
    graph = graph or load_graph(data_root)
    payload = {"articles": sorted(graph.articles), "links": graph.links, "texts": graph.texts}
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8"))


def _model_base_revision(model: Any) -> str:
    metadata = getattr(model, "metadata", {})
    if isinstance(metadata, dict) and isinstance(metadata.get("base_revision"), str):
        return metadata["base_revision"]
    for name in ("base_revision", "resolved_base_revision"):
        value = getattr(model, name, None)
        if isinstance(value, str) and value:
            return value
    raise ValueError("model metadata must expose the immutable base model revision")


def _model_metadata_value(model: Any, key: str, fallback: str) -> str:
    metadata = getattr(model, "metadata", {})
    if isinstance(metadata, dict):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            return value
    return fallback


def _tokenizer_fingerprint(model: Any) -> str:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("training data preparation requires model.tokenizer")
    if hasattr(tokenizer, "get_vocab"):
        vocab = tokenizer.get_vocab()
    else:
        vocab = getattr(tokenizer, "vocab", None)
    if vocab is None:
        raise ValueError("cannot fingerprint tokenizer vocabulary")
    backend = getattr(tokenizer, "backend_tokenizer", None)
    tokenizer_json = backend.to_str() if backend is not None and hasattr(backend, "to_str") else None
    payload = {
        "tokenizer_json": tokenizer_json,
        "vocab": sorted(vocab.items(), key=lambda item: (item[1], item[0])),
        "special_tokens_map": getattr(tokenizer, "special_tokens_map", {}),
    }
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8"))


def _render_state(graph: WikiGraph, current: str, target: str, history: list[str],
                  excerpt_chars: int) -> dict[str, Any]:
    return {
        "current_title": display_title(current),
        "target_title": display_title(target),
        "current_excerpt": graph.texts.get(current, "")[:excerpt_chars],
        "history": [display_title(article) for article in history],
    }


def _template_fingerprint() -> str:
    state = {"current_title": "CURRENT", "target_title": "TARGET",
             "current_excerpt": "", "history": ["CURRENT"]}
    request = make_request(state, ["link_0"], ["ARTICLE"])
    return _sha256_bytes(json.dumps(request, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode("utf-8"))


def _shortest_distances(data_root: Path, graph: WikiGraph,
                        pairs: set[tuple[str, str]]) -> dict[tuple[str, str], int | None]:
    """Read SNAP's precomputed matrix, or use BFS for small custom fixtures."""
    articles_files = sorted((data_root / "raw").rglob("articles.tsv"))
    matrix_files = sorted((data_root / "raw").rglob("shortest-path-distance-matrix.txt"))
    if articles_files and matrix_files:
        ordered_articles: list[str] = []
        with articles_files[0].open("r", encoding="utf-8", errors="replace") as stream:
            for line in stream:
                text = line.strip()
                if text and not text.startswith("#"):
                    ordered_articles.append(decode_id(text.split("\t")[0]))
        indexes = {article: index for index, article in enumerate(ordered_articles)}
        wanted: dict[int, set[int]] = {}
        for start, target in pairs:
            if start not in indexes or target not in indexes:
                raise ValueError(f"task article missing from shortest-path matrix: {start!r}, {target!r}")
            wanted.setdefault(indexes[start], set()).add(indexes[target])
        distances: dict[tuple[str, str], int | None] = {}
        with matrix_files[0].open("r", encoding="utf-8") as stream:
            row_index = 0
            for line in stream:
                text = line.rstrip("\r\n")
                if not text or text.startswith("#"):
                    continue
                if row_index >= len(ordered_articles) or len(text) != len(ordered_articles):
                    raise ValueError("shortest-path matrix dimensions do not match articles.tsv")
                if row_index in wanted:
                    start = ordered_articles[row_index]
                    for target_index in wanted[row_index]:
                        target = ordered_articles[target_index]
                        value = text[target_index]
                        distances[(start, target)] = int(value) if value.isdigit() else None
                row_index += 1
        if row_index != len(ordered_articles):
            raise ValueError("shortest-path matrix row count does not match articles.tsv")
        return distances

    distances = {}
    targets_by_start: dict[str, set[str]] = {}
    for start, target in pairs:
        targets_by_start.setdefault(start, set()).add(target)
    for start, targets in targets_by_start.items():
        found = {start: 0}
        queue = deque([start])
        while queue and any(target not in found for target in targets):
            current = queue.popleft()
            for next_article in graph.links.get(current, []):
                if next_article not in found:
                    found[next_article] = found[current] + 1
                    queue.append(next_article)
        for target in targets:
            distances[(start, target)] = found.get(target)
    return distances


def _preflight_graph(model: Any, graph: WikiGraph, config: Any) -> dict[str, Any]:
    """Bound every candidate pair using exact Kev option and instruction spans."""
    option_counter = getattr(model, "option_token_count", None)
    instruction_counter = getattr(model, "instruction_token_count", None)
    if option_counter is None or instruction_counter is None:
        raise ValueError("model must expose option_token_count and instruction_token_count for preflight")
    instruction_tokens = int(instruction_counter(INSTRUCTIONS))
    option_cache: dict[tuple[str, str], int] = {}
    worst_bound = 0
    worst_source: str | None = None
    checked_pages = 0
    for source in sorted(graph.articles):
        moves = graph.links.get(source, [])
        if len(moves) < 2:
            continue
        span_counts: list[int] = []
        for index, article in enumerate(moves):
            candidate_id = f"link_{index}"
            key = (candidate_id, article)
            if key not in option_cache:
                option_cache[key] = int(option_counter(candidate_id, article))
            span_counts.append(option_cache[key])
        span_counts.sort(reverse=True)
        bound = config.choices.max_state_tokens + instruction_tokens + sum(span_counts[:2])
        checked_pages += 1
        if bound > worst_bound:
            worst_bound = bound
            worst_source = source
        if bound > config.choices.max_row_tokens:
            raise ValueError(
                f"choice token preflight failed at {source}: worst pair bound {bound} "
                f"exceeds max_row_tokens={config.choices.max_row_tokens}")
    return {
        "available": True,
        "checked_pages": checked_pages,
        "checked_options": len(option_cache),
        "instruction_tokens": instruction_tokens,
        "state_budget_tokens": config.choices.max_state_tokens,
        "worst_pair_bound": worst_bound,
        "worst_source": worst_source,
    }


def _stage_bytes(path: Path, content: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=path.name + ".",
                                     suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    return temporary


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def prepare_training(config: Any, model: Any, *, max_paths: int | None = None,
                     force: bool = False) -> dict[str, Any]:
    """Prepare deterministic pair splits and human SFT data; return the manifest."""
    if max_paths is None:
        max_paths = config.data.max_paths
    if max_paths is not None and max_paths < 1:
        raise ValueError("max_paths must be positive when provided")
    data_root = Path(config.data_dir)
    graph = load_graph(data_root)
    paths_file = _raw_paths_file(data_root)
    raw_routes = _read_raw_routes(paths_file)
    selected_ids: list[int] | None = None
    if max_paths is not None:
        raw_routes = sorted(raw_routes, key=lambda item: (
            hashlib.sha256(f"{config.seed}\0{item[0]}".encode()).digest(), item[0]))[:max_paths]
        raw_routes.sort(key=lambda item: item[0])
        selected_ids = [row_id for row_id, _ in raw_routes]

    baseline_path = BASELINE_TASKS
    baseline_tasks = json.loads(baseline_path.read_text(encoding="utf-8"))
    baseline_pairs = {
        (str(item["start"]), str(item["target"])) for item in baseline_tasks
        if str(item["start"]) in graph.articles and str(item["target"]) in graph.articles
    }
    graph_sha256 = _graph_fingerprint(data_root, graph)
    tokenizer_sha256 = _tokenizer_fingerprint(model)
    identity = {
        "schema_version": 1,
        "preprocessing_version": PREPROCESSING_VERSION,
        "seed": config.seed,
        "max_steps": config.rl.max_steps,
        "max_paths": max_paths,
        "selected_source_row_ids": selected_ids,
        "selection_sha256": _sha256_bytes(json.dumps(selected_ids, separators=(",", ":")).encode()),
        "raw_paths_sha256": _sha256_file(paths_file),
        "processed_graph_sha256": graph_sha256,
        "baseline_tasks_sha256": _sha256_file(baseline_path),
        "choice_config": asdict(config.choices),
        "choice_template_sha256": _template_fingerprint(),
        "tokenizer_sha256": tokenizer_sha256,
        "base_checkpoint": _model_metadata_value(model, "checkpoint", config.model.checkpoint),
        "base_model": _model_metadata_value(model, "base", config.model.checkpoint),
        "base_revision": _model_base_revision(model),
    }
    manifest_path = _relative_path(config, "split_manifest")
    output_paths = [_relative_path(config, name) for name in (
        "sft_train", "sft_dev", "rl_train_tasks", "dev_tasks", "test_tasks")]
    if manifest_path.exists() and not force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if all(existing.get(key) == value for key, value in identity.items()):
            for output_path in output_paths:
                expected = existing.get("output_sha256", {}).get(str(output_path))
                if not output_path.is_file() or _sha256_file(output_path) != expected:
                    raise FileExistsError(f"Existing prepared output {output_path} does not match its manifest")
            return existing
        raise FileExistsError(
            f"Existing training manifest at {manifest_path} is incompatible; pass force=True to replace it")
    token_preflight = _preflight_graph(model, graph, config)
    rejection_counts: Counter[str] = Counter()
    rejected_rows: list[dict[str, Any]] = []
    rejected_transitions: Counter[tuple[str, str]] = Counter()
    valid_routes: list[tuple[int, list[str]]] = []
    for row_id, route in raw_routes:
        clicks = len(route) - 1
        if not 1 <= clicks <= config.rl.max_steps:
            rejection_counts["click_count_outside_1_to_max_steps"] += 1
            rejected_rows.append({"source_row_id": row_id, "reason": "click_count_outside_1_to_max_steps"})
            continue
        if route[0] == route[-1]:
            rejection_counts["start_equals_target"] += 1
            rejected_rows.append({"source_row_id": row_id, "reason": "start_equals_target"})
            continue
        reason = None
        for source, target in zip(route, route[1:]):
            if source not in graph.articles or target not in graph.articles:
                reason = "unknown_article"
                break
            if target not in graph.links.get(source, []):
                reason = "illegal_transition"
                rejected_transitions[(source, target)] += 1
                break
        if reason:
            rejection_counts[reason] += 1
            rejected_rows.append({"source_row_id": row_id, "reason": reason})
            continue
        valid_routes.append((row_id, route))

    pair_splits: dict[tuple[str, str], str] = {}
    for start, target in baseline_pairs:
        pair_splits[(start, target)] = "dev"
    for _, route in valid_routes:
        start, target = route[0], route[-1]
        pair_splits.setdefault((start, target), _pair_split(start, target, config.seed))
    for pair in baseline_pairs:
        pair_splits[pair] = "dev"

    train_tasks: set[tuple[str, str]] = set()
    dev_tasks: set[tuple[str, str]] = set(baseline_pairs)
    test_tasks: set[tuple[str, str]] = set()
    for pair, split in pair_splits.items():
        if pair in baseline_pairs:
            continue
        if split == "train":
            train_tasks.add(pair)
        elif split == "dev":
            dev_tasks.add(pair)
        else:
            test_tasks.add(pair)

    output_targets = {
        "train_sft": _relative_path(config, "sft_train"),
        "dev_sft": _relative_path(config, "sft_dev"),
        "train_tasks": _relative_path(config, "rl_train_tasks"),
        "dev_tasks": _relative_path(config, "dev_tasks"),
        "test_tasks": _relative_path(config, "test_tasks"),
    }
    temporary_paths: dict[str, Path] = {}
    sft_streams: dict[str, Any] = {}
    sft_counts = Counter()
    manifest_stage: Path | None = None
    try:
        for split in ("train", "dev"):
            target_path = output_targets[f"{split}_sft"]
            target_path.parent.mkdir(parents=True, exist_ok=True)
            stream = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", newline="\n", dir=target_path.parent,
                prefix=target_path.name + ".", suffix=".tmp", delete=False)
            temporary_paths[f"{split}_sft"] = Path(stream.name)
            sft_streams[split] = stream

        for row_id, route in valid_routes:
            start, target = route[0], route[-1]
            split = pair_splits[(start, target)]
            if split not in sft_streams:
                continue
            trajectory_id = f"human-{row_id:06d}"
            for step, (current, next_article) in enumerate(zip(route, route[1:])):
                moves = graph.links.get(current, [])
                candidate_ids = [f"link_{index}" for index in range(len(moves))]
                try:
                    label_id = candidate_ids[moves.index(next_article)]
                except ValueError as error:
                    raise RuntimeError("validated route lost its demonstrated edge") from error
                state = _render_state(graph, current, target, route[:step + 1],
                                      config.choices.excerpt_chars)
                derived_seed = int.from_bytes(hashlib.sha256(
                    f"{config.seed}\0{row_id}\0{step}".encode()).digest()[:8], "big")
                nodes = build_supervised_nodes(model, config.choices, state, candidate_ids,
                                               moves, label_id, derived_seed)
                for node in nodes:
                    example_id = (f"{trajectory_id}-step-{step}-round-{node['round']}-"
                                  f"node-{node['node']}")
                    example = {
                        "schema_version": 1,
                        "example_id": example_id,
                        "trajectory_id": trajectory_id,
                        "source_row_id": row_id,
                        "pair": {"start": start, "target": target},
                        "source": "human",
                        "step": step,
                        **node,
                    }
                    sft_streams[split].write(json.dumps(
                        example, ensure_ascii=False, separators=(",", ":")) + "\n")
                    sft_counts[split] += 1

        for stream in sft_streams.values():
            stream.close()
        sft_streams.clear()
        task_sets = {"train_tasks": train_tasks, "dev_tasks": dev_tasks,
                     "test_tasks": test_tasks}
        for key, pairs in task_sets.items():
            tasks = [{"start": start, "target": target}
                     for start, target in sorted(pairs)]
            temporary_paths[key] = _stage_bytes(output_targets[key], _json_bytes(tasks))
        pair_assignments = [{"start": start, "target": target, "split": split}
                            for (start, target), split in sorted(pair_splits.items())]
        task_pairs = set(pair_splits)
        shortest_distances = _shortest_distances(data_root, graph, task_pairs)
        source_rows_by_pair: dict[tuple[str, str], list[int]] = {}
        for row_id, route in valid_routes:
            source_rows_by_pair.setdefault((route[0], route[-1]), []).append(row_id)
        for assignment in pair_assignments:
            pair = (assignment["start"], assignment["target"])
            assignment["source"] = ["baseline"] if pair in baseline_pairs else ["human"]
            if source_rows_by_pair.get(pair) and pair in baseline_pairs:
                assignment["source"].append("human")
            assignment["source_row_ids"] = source_rows_by_pair.get(pair, [])
            assignment["shortest_distance"] = shortest_distances[pair]
        output_hashes = {
            str(output_targets[key]): _sha256_file(temp_path)
            for key, temp_path in temporary_paths.items()
        }
        manifest: dict[str, Any] = {
            **identity,
            "split_fractions": {"train": 0.8, "dev": 0.1, "test": 0.1},
            "pair_assignments": pair_assignments,
            "source_row_ids": [row_id for row_id, _ in valid_routes],
            "rejected_source_rows": rejected_rows,
            "rejected_transitions": [
                {"source": source, "target": target, "count": count}
                for (source, target), count in sorted(rejected_transitions.items())
            ],
            "rejection_counts": dict(sorted(rejection_counts.items())),
            "valid_route_count": len(valid_routes),
            "sft_example_counts": dict(sft_counts),
            "task_counts": {"train": len(train_tasks), "dev": len(dev_tasks),
                            "test": len(test_tasks)},
            "token_preflight": token_preflight,
            "output_sha256": output_hashes,
        }
        manifest_stage = _stage_bytes(manifest_path, _json_bytes(manifest))
        for key, temp_path in temporary_paths.items():
            temp_path.replace(output_targets[key])
        manifest_stage.replace(manifest_path)
        manifest_stage = None
        return manifest
    finally:
        for stream in sft_streams.values():
            stream.close()
        for path in temporary_paths.values():
            path.unlink(missing_ok=True)
        if manifest_stage is not None:
            manifest_stage.unlink(missing_ok=True)


def validate_prepared_data(config: Any, model: Any | None = None) -> dict[str, Any]:
    """Verify prepared files, preprocessing identity, and pair split isolation."""
    manifest_path = _relative_path(config, "split_manifest")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared-data manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("Prepared-data manifest must be a JSON object")
    expected_values = {
        "schema_version": 1,
        "preprocessing_version": PREPROCESSING_VERSION,
        "seed": config.seed,
        "max_steps": config.rl.max_steps,
        "choice_config": asdict(config.choices),
        "choice_template_sha256": _template_fingerprint(),
        "processed_graph_sha256": _graph_fingerprint(Path(config.data_dir)),
        "raw_paths_sha256": _sha256_file(_raw_paths_file(Path(config.data_dir))),
        "baseline_tasks_sha256": _sha256_file(BASELINE_TASKS),
    }
    for key, expected in expected_values.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Prepared-data manifest {key} does not match the current configuration or data")

    token_preflight = manifest.get("token_preflight")
    if not isinstance(token_preflight, dict) or token_preflight.get("available") is not True:
        raise ValueError("Prepared-data manifest is missing successful tokenizer preflight metadata")

    output_paths = {
        "sft_train": _relative_path(config, "sft_train"),
        "sft_dev": _relative_path(config, "sft_dev"),
        "rl_train_tasks": _relative_path(config, "rl_train_tasks"),
        "dev_tasks": _relative_path(config, "dev_tasks"),
        "test_tasks": _relative_path(config, "test_tasks"),
    }
    output_hashes = manifest.get("output_sha256")
    if not isinstance(output_hashes, dict):
        raise ValueError("Prepared-data manifest is missing output hashes")
    for path in output_paths.values():
        expected_hash = output_hashes.get(str(path))
        if not path.is_file() or not isinstance(expected_hash, str):
            raise ValueError(f"Prepared-data output is missing or absent from manifest: {path}")
        if _sha256_file(path) != expected_hash:
            raise ValueError(f"Prepared-data output hash mismatch: {path}")

    if model is not None:
        if _model_base_revision(model) != manifest.get("base_revision"):
            raise ValueError("Prepared-data base model revision does not match the loaded model")
        if _tokenizer_fingerprint(model) != manifest.get("tokenizer_sha256"):
            raise ValueError("Prepared-data tokenizer fingerprint does not match the loaded model")

    assignments = manifest.get("pair_assignments")
    if not isinstance(assignments, list):
        raise ValueError("Prepared-data manifest pair_assignments must be an array")
    assignment_split: dict[tuple[str, str], str] = {}
    expected_pairs = {"train": set(), "dev": set(), "test": set()}
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("Every pair assignment must be an object")
        start, target, split = (assignment.get("start"), assignment.get("target"),
                                assignment.get("split"))
        if not all(isinstance(value, str) for value in (start, target)):
            raise ValueError("Pair assignment start and target must be strings")
        if split not in expected_pairs:
            raise ValueError(f"Invalid pair assignment split: {split!r}")
        pair = (start, target)
        if pair in assignment_split:
            raise ValueError(f"Duplicate pair assignment: {pair!r}")
        assignment_split[pair] = split
        expected_pairs[split].add(pair)

    task_files = {
        "train": output_paths["rl_train_tasks"],
        "dev": output_paths["dev_tasks"],
        "test": output_paths["test_tasks"],
    }
    observed_pairs: dict[str, set[tuple[str, str]]] = {}
    for split, path in task_files.items():
        items = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(items, list):
            raise ValueError(f"{split} task file must contain a JSON array")
        pairs: set[tuple[str, str]] = set()
        for item in items:
            if not isinstance(item, dict):
                raise ValueError(f"Every {split} task must be an object")
            start, target = item.get("start"), item.get("target")
            if not isinstance(start, str) or not isinstance(target, str):
                raise ValueError(f"Every {split} task needs string start and target IDs")
            pair = (start, target)
            if pair in pairs:
                raise ValueError(f"Duplicate {split} task pair: {pair!r}")
            if assignment_split.get(pair) != split:
                raise ValueError(f"Task pair {pair!r} is not assigned to {split}")
            pairs.add(pair)
        if pairs != expected_pairs[split]:
            raise ValueError(f"{split} task file does not match its manifest pair assignments")
        observed_pairs[split] = pairs
    if (observed_pairs["train"] & observed_pairs["dev"]
            or observed_pairs["train"] & observed_pairs["test"]
            or observed_pairs["dev"] & observed_pairs["test"]):
        raise ValueError("Prepared task files contain cross-split pair overlap")

    for split, path in (("train", output_paths["sft_train"]),
                        ("dev", output_paths["sft_dev"])):
        with path.open("r", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                example = json.loads(line)
                pair_info = example.get("pair") if isinstance(example, dict) else None
                if not isinstance(pair_info, dict):
                    raise ValueError(f"Invalid {split} SFT example at {path}:{line_number}: missing pair")
                pair = (pair_info.get("start"), pair_info.get("target"))
                if assignment_split.get(pair) != split:
                    raise ValueError(f"SFT example pair {pair!r} is not assigned to {split}")
    return manifest
