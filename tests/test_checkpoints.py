import hashlib
import importlib.util
import os
import random
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from wiki_speedrun.training.checkpoints import (
    file_fingerprints,
    load_training_state,
    resolve_checkpoint,
    save_training_checkpoint,
    write_json_atomic,
)
import wiki_speedrun.training.checkpoints as checkpoints


class CheckpointFileTests(unittest.TestCase):
    def test_atomic_pointer_resolves_update_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            update = root / "update-000001"
            update.mkdir()
            pointer = root / "latest.json"
            write_json_atomic(pointer, {"schema_version": 1, "checkpoint": update.name})
            self.assertEqual(resolve_checkpoint(pointer), update)

    def test_run_directory_resolves_latest_pointer(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            update = root / "update-000004"
            update.mkdir()
            write_json_atomic(root / "latest.json", {"schema_version": 1,
                                                        "checkpoint": update.name})
            self.assertEqual(resolve_checkpoint(root), update)

    def test_required_input_fingerprint_and_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "input.json"
            path.write_text("{}\n", encoding="utf-8")
            fingerprints = file_fingerprints([path])
            self.assertEqual(fingerprints[str(path.resolve())],
                             hashlib.sha256(path.read_bytes()).hexdigest())
            with self.assertRaises(FileNotFoundError):
                file_fingerprints([path.parent / "missing.json"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "training extra is not installed")
    def test_resume_restores_the_same_next_optimizer_update(self):
        import torch

        class SmallModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)

            def forward(self, value):
                return self.linear(value)

            def save_checkpoint(self, directory):
                torch.save(self.state_dict(), Path(directory) / "weights.pt")

        with tempfile.TemporaryDirectory() as temp:
            previous_cwd = Path.cwd()
            os.chdir(temp)
            try:
                torch.manual_seed(23)
                model = SmallModel()
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
                x = torch.tensor([[0.25, -0.75]])
                loss = model(x).square().mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                random.seed(314)
                run_dir = Path("runs/run-a")
                run_dir.mkdir(parents=True)
                checkpoint = save_training_checkpoint(
                    model=model, optimizer=optimizer, config={"seed": 23},
                    run_dir=run_dir, run_id="run-a", phase="rl", update=1,
                    cursor={"next_update": 2}, best_score=None, data_hashes={},
                )
                expected_sample = torch.rand(1)
                expected_tasks = random.sample(list(range(10)), 4)
                next_loss = model(x).square().mean()
                next_loss.backward()
                optimizer.step()
                expected_weights = {key: value.detach().clone()
                                    for key, value in model.state_dict().items()}

                resumed_model = SmallModel()
                resumed_model.load_state_dict(torch.load(checkpoint / "weights.pt",
                                                         weights_only=True))
                resumed_optimizer = torch.optim.AdamW(resumed_model.parameters(), lr=1e-2)
                state = load_training_state(checkpoint, config={"seed": 23},
                                            optimizer=resumed_optimizer,
                                            expected_data_hashes={}, run_dir=run_dir)
                self.assertEqual(state["completed_update"], 1)
                self.assertTrue(torch.equal(torch.rand(1), expected_sample))
                self.assertEqual(random.sample(list(range(10)), 4), expected_tasks)
                resumed_loss = resumed_model(x).square().mean()
                resumed_loss.backward()
                resumed_optimizer.step()
                for key, value in resumed_model.state_dict().items():
                    self.assertTrue(torch.equal(value, expected_weights[key]))
            finally:
                os.chdir(previous_cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "training extra is not installed")
    def test_retention_keeps_recent_and_best_then_releases_old_best(self):
        import torch

        class SmallModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1)

            def save_checkpoint(self, directory):
                torch.save(self.state_dict(), Path(directory) / "weights.pt")

        with tempfile.TemporaryDirectory() as temp:
            previous_cwd = Path.cwd()
            os.chdir(temp)
            try:
                model = SmallModel()
                optimizer = torch.optim.AdamW(model.parameters())
                run_dir = Path("runs/run-retention")
                run_dir.mkdir(parents=True)
                saved = {}
                for update in range(1, 7):
                    saved[update] = save_training_checkpoint(
                        model=model, optimizer=optimizer, config={"seed": 7},
                        run_dir=run_dir, run_id="run-retention", phase="rl",
                        update=update, cursor={"next_update": update + 1},
                        best_score=None, data_hashes={}, make_best=(update == 1),
                    )
                root = Path("checkpoints/run-retention")
                self.assertFalse(saved[2].exists())
                self.assertTrue(saved[1].is_dir())  # best, despite being older
                self.assertTrue(saved[5].is_dir())
                self.assertTrue(saved[6].is_dir())
                self.assertEqual(resolve_checkpoint(root / "latest.json"), saved[6])
                self.assertEqual(resolve_checkpoint(root / "best.json"), saved[1])
                resumed = SmallModel()
                resumed_optimizer = torch.optim.AdamW(resumed.parameters())
                state = load_training_state(root / "latest.json", config={"seed": 7},
                                            optimizer=resumed_optimizer,
                                            expected_data_hashes={}, run_dir=run_dir)
                self.assertEqual(state["completed_update"], 6)

                saved[7] = save_training_checkpoint(
                    model=model, optimizer=optimizer, config={"seed": 7},
                    run_dir=run_dir, run_id="run-retention", phase="rl",
                    update=7, cursor={"next_update": 8}, best_score=None,
                    data_hashes={}, make_best=True,
                )
                self.assertFalse(saved[1].exists())  # replaced best no longer pins it
                self.assertTrue(saved[6].is_dir())
                self.assertTrue(saved[7].is_dir())
                self.assertEqual(resolve_checkpoint(root / "best.json"), saved[7])
                state = load_training_state(root / "best.json", config={"seed": 7},
                                            expected_data_hashes={}, run_dir=run_dir)
                self.assertEqual(state["completed_update"], 7)
            finally:
                os.chdir(previous_cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "training extra is not installed")
    def test_retention_preserves_untrusted_and_unfinished_entries(self):
        import torch

        class SmallModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def save_checkpoint(self, directory):
                torch.save(self.state_dict(), Path(directory) / "weights.pt")

        with tempfile.TemporaryDirectory() as temp:
            previous_cwd = Path.cwd()
            os.chdir(temp)
            try:
                model = SmallModel()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                run_dir = Path("runs/run-safe")
                run_dir.mkdir(parents=True)
                root = Path("checkpoints/run-safe")
                root.mkdir(parents=True)
                foreign = root / "update-000001"
                foreign.mkdir()
                (foreign / "manifest.json").write_text(
                    '{"run_id":"another-run","completed_update":1}', encoding="utf-8")
                unmanifested = root / "update-000002"
                unmanifested.mkdir()
                temporary = root / ".update-000003.tmp"
                temporary.mkdir()
                external = Path(temp) / "external"
                external.mkdir()
                link = root / "update-000004"
                link.symlink_to(external, target_is_directory=True)
                for update in range(5, 11):
                    save_training_checkpoint(
                        model=model, optimizer=optimizer, config={"seed": 1},
                        run_dir=run_dir, run_id="run-safe", phase="rl", update=update,
                        cursor={}, best_score=None, data_hashes={},
                    )
                self.assertTrue(foreign.is_dir())
                self.assertTrue(unmanifested.is_dir())
                self.assertTrue(temporary.is_dir())
                self.assertTrue(link.is_symlink())
            finally:
                os.chdir(previous_cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "training extra is not installed")
    def test_retention_is_skipped_after_export_or_pointer_failure(self):
        import torch

        class SmallModel(torch.nn.Module):
            def __init__(self, fail=False):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))
                self.fail = fail

            def save_checkpoint(self, directory):
                if self.fail:
                    raise RuntimeError("export failed")
                torch.save(self.state_dict(), Path(directory) / "weights.pt")

        with tempfile.TemporaryDirectory() as temp:
            previous_cwd = Path.cwd()
            os.chdir(temp)
            try:
                model = SmallModel()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                run_dir = Path("runs/run-failure")
                run_dir.mkdir(parents=True)
                saved = []
                for update in range(1, 4):
                    saved.append(save_training_checkpoint(
                        model=model, optimizer=optimizer, config={"seed": 1},
                        run_dir=run_dir, run_id="run-failure", phase="rl", update=update,
                        cursor={}, best_score=None, data_hashes={},
                    ))
                existing = {path.name for path in Path("checkpoints/run-failure").iterdir()
                            if path.is_dir() and not path.is_symlink()}
                with self.assertRaises(RuntimeError):
                    save_training_checkpoint(
                        model=SmallModel(fail=True), optimizer=optimizer, config={"seed": 1},
                        run_dir=run_dir, run_id="run-failure", phase="rl", update=4,
                        cursor={}, best_score=None, data_hashes={},
                    )
                after_export_failure = {
                    path.name for path in Path("checkpoints/run-failure").iterdir()
                    if path.is_dir() and not path.is_symlink()
                }
                self.assertEqual(after_export_failure, existing)

                real_write = checkpoints.write_json_atomic

                def fail_latest(path, payload):
                    if Path(path).name == "latest.json":
                        raise OSError("pointer failed")
                    return real_write(path, payload)

                with mock.patch.object(checkpoints, "write_json_atomic", side_effect=fail_latest):
                    with self.assertRaises(OSError):
                        save_training_checkpoint(
                            model=model, optimizer=optimizer, config={"seed": 1},
                            run_dir=run_dir, run_id="run-failure", phase="rl", update=4,
                            cursor={}, best_score=None, data_hashes={},
                        )
                after_pointer_failure = {
                    path.name for path in Path("checkpoints/run-failure").iterdir()
                    if path.is_dir() and not path.is_symlink()
                }
                self.assertEqual(after_pointer_failure, existing | {"update-000004"})
            finally:
                os.chdir(previous_cwd)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "training extra is not installed")
    def test_invalid_pointers_skip_pruning_and_unpublished_latest_keeps_current(self):
        import torch

        class SmallModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1))

            def save_checkpoint(self, directory):
                torch.save(self.state_dict(), Path(directory) / "weights.pt")

        with tempfile.TemporaryDirectory() as temp:
            previous_cwd = Path.cwd()
            os.chdir(temp)
            try:
                model = SmallModel()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                for run_id, invalid_pointer in (
                    ("run-malformed", "malformed"),
                    ("run-stale", "stale"),
                    ("run-broken-link", "broken-link"),
                ):
                    run_dir = Path("runs") / run_id
                    run_dir.mkdir(parents=True)
                    for update in range(1, 4):
                        save_training_checkpoint(
                            model=model, optimizer=optimizer, config={"seed": 1},
                            run_dir=run_dir, run_id=run_id, phase="rl", update=update,
                            cursor={}, best_score=None, data_hashes={},
                        )
                    root = Path("checkpoints") / run_id
                    before = {f"update-{update:06d}" for update in (2, 3)}
                    latest = root / "latest.json"
                    latest.unlink()
                    if invalid_pointer == "malformed":
                        latest.write_text("{not json", encoding="utf-8")
                    elif invalid_pointer == "stale":
                        write_json_atomic(latest, {"checkpoint": "update-009999"})
                    else:
                        latest.symlink_to(root / "missing-pointer-target.json")
                    save_training_checkpoint(
                        model=model, optimizer=optimizer, config={"seed": 1},
                        run_dir=run_dir, run_id=run_id, phase="rl", update=4,
                        cursor={}, best_score=None, data_hashes={},
                        make_latest=False, make_best=False,
                    )
                    remaining = {path.name for path in root.iterdir()
                                 if path.is_dir() and not path.is_symlink()}
                    self.assertEqual(remaining, before | {"update-000004"})

                # A valid latest pointer can remain unchanged while the just-saved
                # checkpoint is still protected, even outside the numeric tail.
                run_id = "run-no-latest"
                run_dir = Path("runs") / run_id
                run_dir.mkdir(parents=True)
                config = {"seed": 1, "save": {"keep_last_checkpoints": 1}}
                for update in (5, 6, 7):
                    save_training_checkpoint(
                        model=model, optimizer=optimizer, config=config,
                        run_dir=run_dir, run_id=run_id, phase="rl", update=update,
                        cursor={}, best_score=None, data_hashes={},
                    )
                saved = save_training_checkpoint(
                    model=model, optimizer=optimizer, config=config,
                    run_dir=run_dir, run_id=run_id, phase="rl", update=2,
                    cursor={}, best_score=None, data_hashes={},
                    make_latest=False, make_best=False,
                )
                self.assertTrue(saved.is_dir())
                self.assertEqual(resolve_checkpoint(Path("checkpoints") / run_id),
                                 Path("checkpoints") / run_id / "update-000007")
            finally:
                os.chdir(previous_cwd)


if __name__ == "__main__":
    unittest.main()
