# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

import json
import os
from pathlib import Path

import pytest

from coreason_model_foundry.curator.deduplicator import SemDeDup
from coreason_model_foundry.curator.formatter import DataFormatter
from coreason_model_foundry.curator.main import Curator
from coreason_model_foundry.curator.resolver import DataResolver
from coreason_model_foundry.schemas import (
    ComputeConfig,
    DatasetConfig,
    MethodConfig,
    MethodType,
    TrainingManifest,
)

# --- Resolver Edge Cases ---


def test_resolver_corrupt_json(tmp_path: Path) -> None:
    resolver = DataResolver()

    # Create corrupt file dynamically
    file_path = tmp_path / "corrupt.json"
    with open(file_path, "w") as f:
        f.write('{ "this": "is broken JSON...')

    with pytest.raises(json.JSONDecodeError):
        # Pass the absolute path. DataResolver logic will replace 'synthesis://' and then treat remainder as path.
        resolver.resolve(f"synthesis://{file_path}")


def test_resolver_dict_not_list(tmp_path: Path) -> None:
    resolver = DataResolver()

    file_path = tmp_path / "dict_not_list.json"
    with open(file_path, "w") as f:
        json.dump({"info": "This is a dict, not a list", "data": []}, f)

    with pytest.raises(ValueError, match="must be a list of records"):
        resolver.resolve(f"synthesis://{file_path}")


def test_resolver_fallback_logic_no_suffix() -> None:
    """
    Tests the fallback logic where:
    1. The path is interpreted as relative to CWD.
    2. The path has no suffix, so .json is appended.
    """
    resolver = DataResolver()

    # Create a temp file in the current working directory
    filename = "temp_test_data_fallback"
    # Ensure cleanup
    try:
        with open(f"{filename}.json", "w") as f:
            json.dump([{"test": "ok"}], f)

        # Call resolve with just the filename (no extension, no path)
        # This triggers:
        # 1. Path(filename).exists() -> False (because missing .json suffix)
        # 2. Path(filename).suffix -> False -> append .json -> Path("temp_test_data_fallback.json")
        # BUT wait, the code checks:
        # file_path = Path(resource_name)
        # if not file_path.suffix: file_path = file_path.with_suffix(".json")
        # If that exists, it uses it.
        # So "temp_test_data_fallback" -> "temp_test_data_fallback.json" -> exists in CWD.
        # This actually hits the FIRST block if run from CWD.

        # To hit the FALLBACK block (lines 65+), the first block must fail.
        # The first block uses Path(resource_name).
        # If resource_name is "temp_test_data_fallback", Path resolves to ./temp_test_data_fallback.json which exists.

        # To hit fallback, we need a case where Path(resource_name) relative resolution fails,
        # but Path.cwd() / resource_name succeeds?
        # Actually Path("foo") IS relative to CWD.
        # The fallback `Path.cwd() / resource_name` is redundant if `Path(resource_name)` is already checking CWD.
        # However, checking `if not file_path.exists()` happens after suffix modification.

        # Let's try to pass a path that assumes a different base but forces the fallback logic?
        # Actually, the fallback block is specifically:
        # if not file_path.exists():
        #     file_path = Path.cwd() / resource_name

        # If I am in a subdirectory, say `tests/`, and I pass a file that exists in `root`,
        # `Path("file.json")` might look in `tests/`. `Path.cwd()` would be `tests/`.
        # Wait, `Path.cwd()` is the process CWD. `Path("rel")` is relative to CWD.
        # They are effectively the same unless `resource_name` is absolute.

        # The logic seems to be:
        # 1. Check relative to CWD (or absolute if provided).
        # 2. If fail, explicitly construct absolute path from CWD.
        # This seems redundant, but to cover it, we just need to make sure we go through it.
        # Maybe the first `exists()` check fails because we didn't add .json yet?
        # No, the code adds .json BEFORE the first `exists()` check.

        # Let's verify the code:
        # file_path = Path(resource_name)
        # if not file_path.suffix: file_path = file_path.with_suffix(".json")
        # if not file_path.exists(): ...

        # So if I pass "foo", it checks "foo.json". If "foo.json" exists, it skips fallback.
        # To hit fallback, "foo.json" must NOT exist, but `Path.cwd() / "foo"` (plus .json) MUST exist.
        # But `Path("foo.json").resolve()` IS `Path.cwd() / "foo.json"`.
        # So the fallback is logically unreachable unless `Path()` behaves weirdly or `resource_name` is weird.

        # EXCEPT: what if `resource_name` is absolute but missing? No.
        # What if we change the CWD inside the test?
        # If I chdir to `/tmp`, `Path("foo.json")` looks in `/tmp`.
        # `Path.cwd()` is `/tmp`.

        # Perhaps the fallback was intended for when the app is run from a different dir than the file?
        # The only way to hit lines 67-68 (inside the fallback block) is if:
        # 1. First `file_path.exists()` returns False.
        # 2. We enter the block.
        # 3. `resource_name` has no suffix.

        # So we need a file that `Path(resource_name)` DOES NOT see, but `Path.cwd() / resource_name` DOES see?
        # That's impossible for relative paths.
        # Is it possible the first block checks valid path but maybe without the suffix adjustment working
        # for some reason? No.

        # Actually, maybe the redundancy IS the issue for coverage.
        # But I can try to trigger it by providing a path that fails the first check.
        # If I provide a path that doesn't exist, it enters the block.
        # Then it tries to construct it.
        # Then it checks if THAT exists.

        # If I provide a non-existent file, it enters the block, does the logic, fails the second check,
        # and raises FileNotFoundError (which covers the exception raise).
        # AND if I provide a non-existent file WITHOUT suffix, it enters `if not file_path.suffix:` inside the fallback.
        # So simply testing for FileNotFoundError with a suffix-less name should cover those lines!

        with pytest.raises(FileNotFoundError):
            resolver.resolve("synthesis://non_existent_file_no_suffix")

    finally:
        if os.path.exists(f"{filename}.json"):
            os.remove(f"{filename}.json")


# --- Formatter Edge Cases ---


def test_formatter_non_string_values() -> None:
    data = [{"instruction": "Calculate", "input": 123, "output": 456}]
    formatted = DataFormatter.format_and_validate(data, MethodType.DORA)
    assert formatted[0]["input"] == 123
    assert formatted[0]["output"] == 456


def test_formatter_extra_fields() -> None:
    data = [{"instruction": "i", "input": "in", "output": "out", "extra_field": "ignore_me"}]
    formatted = DataFormatter.format_and_validate(data, MethodType.DORA)
    assert "extra_field" not in formatted[0]
    assert "instruction" in formatted[0]


def test_formatter_empty_required_fields() -> None:
    data = [{"instruction": "", "input": "", "output": ""}]
    formatted = DataFormatter.format_and_validate(data, MethodType.DORA)
    assert formatted[0]["instruction"] == ""


def test_formatter_unknown_method() -> None:
    """Test handling of an unknown method type."""
    data = [{"instruction": "i", "output": "o"}]
    # Force pass a string that isn't a valid Enum member to bypass static type checkers
    # or just use a mock if runtime allows.
    # In python, we can just pass a string if the type hint is ignored at runtime.

    with pytest.raises(ValueError, match="Unknown format requirement"):
        DataFormatter.format_and_validate(data, "UNKNOWN_METHOD")  # type: ignore


# --- SemDeDup Edge Cases ---


def test_sem_dedup_long_text() -> None:
    long_text = "word " * 1000
    data = [
        {"instruction": "long", "input": long_text, "output": "out"},
        {"instruction": "long", "input": long_text, "output": "out"},
    ]
    deduper = SemDeDup(model_name="all-MiniLM-L6-v2", threshold=0.99)
    pruned = deduper.prune(data, key_fields=["instruction", "input"])
    assert len(pruned) == 1


def test_sem_dedup_threshold_exact_match() -> None:
    data = [
        {"text": "Hello world"},
        {"text": "Hello world."},
    ]
    deduper = SemDeDup(model_name="all-MiniLM-L6-v2", threshold=0.99999)
    pruned = deduper.prune(data, key_fields=["text"])
    assert len(pruned) == 2


def test_sem_dedup_empty_data() -> None:
    """Test pruning an empty list."""
    deduper = SemDeDup()
    assert deduper.prune([], key_fields=["text"]) == []


# --- Complex Scenario ---


@pytest.fixture
def dirty_dataset_path(tmp_path: Path) -> str:
    """Creates a temporary dirty dataset file."""
    data = [
        {"instruction": "Fix me", "input": "bug", "output": "fixed", "extra": "junk"},
        {"instruction": "Fix me", "input": "bug", "output": "fixed", "extra": "junk"},
        {"instruction": "Fix me", "input": "bug ", "output": "fixed"},
        {"instruction": "Count", "input": 5, "output": 10},
        {"instruction": "Count", "input": 5, "output": 10},
    ]

    file_path = tmp_path / "dirty_data.json"
    with open(file_path, "w") as f:
        json.dump(data, f)

    return str(file_path)


def test_curator_complex_dirty_dataset(dirty_dataset_path: str) -> None:
    manifest = TrainingManifest(
        job_id="complex-job",
        base_model="test-model",
        method_config=MethodConfig(type=MethodType.DORA, rank=16, alpha=32, target_modules=["all"]),
        dataset=DatasetConfig(ref=f"synthesis://{dirty_dataset_path}", sem_dedup=True, dedup_threshold=0.95),
        compute=ComputeConfig(batch_size=1, grad_accum=1, context_window=1024),
    )

    curator = Curator(manifest)
    final_data = curator.prepare_dataset()

    assert len(final_data) == 2
    for item in final_data:
        assert set(item.keys()) == {"instruction", "input", "output"}
        assert "extra" not in item
