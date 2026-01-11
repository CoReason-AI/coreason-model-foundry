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
        # If we pass just the filename, it looks in CWD. If we pass abs path, it should work.
        resolver.resolve(f"synthesis://{file_path}")


def test_resolver_dict_not_list(tmp_path: Path) -> None:
    resolver = DataResolver()

    file_path = tmp_path / "dict_not_list.json"
    with open(file_path, "w") as f:
        json.dump({"info": "This is a dict, not a list", "data": []}, f)

    with pytest.raises(ValueError, match="must be a list of records"):
        resolver.resolve(f"synthesis://{file_path}")


# --- Formatter Edge Cases ---


def test_formatter_non_string_values() -> None:
    """Test that non-string values are preserved or handled."""
    data = [{"instruction": "Calculate", "input": 123, "output": 456}]
    # SFT format usually expects strings, but let's see if our formatter enforces it
    # Currently, it just passes values through.
    # If downstream components (Unsloth) require strings, we might want to cast them here.
    # For now, we assume strict adherence to "garbage in, garbage out" unless we decided to cast.
    # Let's verify they come out as entered.

    formatted = DataFormatter.format_and_validate(data, MethodType.DORA)
    assert formatted[0]["input"] == 123
    assert formatted[0]["output"] == 456


def test_formatter_extra_fields() -> None:
    """Test that extra fields are dropped."""
    data = [{"instruction": "i", "input": "in", "output": "out", "extra_field": "ignore_me"}]
    formatted = DataFormatter.format_and_validate(data, MethodType.DORA)
    assert "extra_field" not in formatted[0]
    assert "instruction" in formatted[0]


def test_formatter_empty_required_fields() -> None:
    """Test that empty strings are accepted (semantics are not validated, only structure)."""
    data = [{"instruction": "", "input": "", "output": ""}]
    formatted = DataFormatter.format_and_validate(data, MethodType.DORA)
    assert formatted[0]["instruction"] == ""


# --- SemDeDup Edge Cases ---


def test_sem_dedup_long_text() -> None:
    """Test behavior with text longer than standard model context (usually 256/512 tokens)."""
    long_text = "word " * 1000  # ~1000 words, definitely > 512 tokens
    data = [
        {"instruction": "long", "input": long_text, "output": "out"},
        {"instruction": "long", "input": long_text, "output": "out"},  # Duplicate
    ]

    deduper = SemDeDup(model_name="all-MiniLM-L6-v2", threshold=0.99)
    # This should not crash. SentenceTransformers usually truncates.
    pruned = deduper.prune(data, key_fields=["instruction", "input"])

    assert len(pruned) == 1


def test_sem_dedup_threshold_exact_match() -> None:
    """Test strict threshold."""
    data = [
        {"text": "Hello world"},
        {"text": "Hello world."},  # Note the period
    ]

    # 1.0 threshold typically means almost exact match (ignoring float precision)
    # Embeddings for "Hello world" and "Hello world." are very close but not 1.0
    deduper = SemDeDup(model_name="all-MiniLM-L6-v2", threshold=0.99999)

    pruned = deduper.prune(data, key_fields=["text"])

    # Expect 2 clusters because they differ slightly and threshold is super high
    assert len(pruned) == 2


# --- Complex Scenario ---


@pytest.fixture
def dirty_dataset_path(tmp_path: Path) -> str:
    """Creates a temporary dirty dataset file."""
    data = [
        # Valid Item 1
        {"instruction": "Fix me", "input": "bug", "output": "fixed", "extra": "junk"},
        # Valid Item 1 Duplicate (Exact)
        {"instruction": "Fix me", "input": "bug", "output": "fixed", "extra": "junk"},
        # Valid Item 1 Duplicate (Semantic - very close)
        # "Fix me" vs "Fix me" and "bug" vs "bug " (trailing space) -> High similarity
        {"instruction": "Fix me", "input": "bug ", "output": "fixed"},
        # Item 2 (Mixed types)
        {"instruction": "Count", "input": 5, "output": 10},
        # Item 2 Duplicate
        {"instruction": "Count", "input": 5, "output": 10},
    ]

    file_path = tmp_path / "dirty_data.json"
    with open(file_path, "w") as f:
        json.dump(data, f)

    return str(file_path)


def test_curator_complex_dirty_dataset(dirty_dataset_path: str) -> None:
    """
    Complex Scenario: "The Dirty Dataset"
    - Reads from file
    - Strips extra keys
    - Handles mixed types (int vs string)
    - Deduplicates semantically
    """
    manifest = TrainingManifest(
        job_id="complex-job",
        base_model="test-model",
        method_config=MethodConfig(type=MethodType.DORA, rank=16, alpha=32, target_modules=["all"]),
        dataset=DatasetConfig(
            ref=f"synthesis://{dirty_dataset_path}",
            sem_dedup=True,
            dedup_threshold=0.95,  # Should catch "bug" vs "bug "
        ),
        compute=ComputeConfig(batch_size=1, grad_accum=1, context_window=1024),
    )

    curator = Curator(manifest)
    final_data = curator.prepare_dataset()

    # Expected Outcome:
    # 1. "Fix me" cluster:
    #    - Original
    #    - Exact duplicate
    #    - Semantic duplicate ("bug ")
    #    -> Should result in 1 representative.
    # 2. "Count" cluster:
    #    - Original (ints)
    #    - Duplicate (ints)
    #    -> Should result in 1 representative.
    # Total = 2 items.

    assert len(final_data) == 2

    # Verify structure is clean
    for item in final_data:
        assert set(item.keys()) == {"instruction", "input", "output"}
        assert "extra" not in item
