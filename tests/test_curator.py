# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry


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


@pytest.fixture
def base_manifest() -> TrainingManifest:
    return TrainingManifest(
        publish_target=None,
        job_id="test-curator-job",
        base_model="test-model",
        method_config=MethodConfig(
            type=MethodType.DORA,
            rank=16,
            alpha=32,
            target_modules=["q_proj"],
            strict_hardware_check=False,
        ),
        dataset=DatasetConfig(ref="synthesis://tests/data/duplicates.json", sem_dedup=True, dedup_threshold=0.99),
        compute=ComputeConfig(quantization="4bit", batch_size=1, grad_accum=1, context_window=1024),
    )


def test_resolver_local_file() -> None:
    resolver = DataResolver()
    # Assuming tests/data/duplicates.json exists (created in setup)
    data = resolver.resolve("synthesis://tests/data/duplicates.json")
    assert isinstance(data, list)
    assert len(data) == 3


def test_resolver_invalid_scheme() -> None:
    resolver = DataResolver()
    with pytest.raises(ValueError, match="Unsupported URI scheme"):
        resolver.resolve("http://google.com/data.json")


def test_resolver_missing_file() -> None:
    resolver = DataResolver()
    with pytest.raises(FileNotFoundError):
        resolver.resolve("synthesis://non_existent_file.json")


def test_formatter_sft_valid() -> None:
    data = [
        {"instruction": "i1", "input": "in1", "output": "o1"},
        {"instruction": "i2", "output": "o2"},  # missing input is allowed (defaults to "")
    ]
    formatted = DataFormatter.format_and_validate(data, MethodType.DORA)
    assert len(formatted) == 2
    assert formatted[0]["input"] == "in1"
    assert formatted[1]["input"] == ""


def test_formatter_sft_invalid() -> None:
    data = [{"input": "in1", "output": "o1"}]  # missing instruction
    with pytest.raises(ValueError, match="missing required keys"):
        DataFormatter.format_and_validate(data, MethodType.DORA)


def test_formatter_orpo_valid() -> None:
    data = [{"prompt": "p1", "chosen": "c1", "rejected": "r1"}]
    formatted = DataFormatter.format_and_validate(data, MethodType.ORPO)
    assert len(formatted) == 1


def test_sem_dedup_logic() -> None:
    """
    Test SemDeDup using actual sentence-transformers if possible,
    or mock if we want to avoid heavy compute in unit tests.
    Since we added the dependency, we can run it.
    """
    # Create duplicative data
    data = [
        {"text": "Hello world"},
        {"text": "Hello world"},  # Exact duplicate
        {"text": "Different text"},
    ]

    deduper = SemDeDup(model_name="all-MiniLM-L6-v2", threshold=0.99)
    pruned = deduper.prune(data, key_fields=["text"])

    assert len(pruned) == 2
    # Should keep one "Hello world" and one "Different text"


def test_curator_full_flow_dora(base_manifest: TrainingManifest) -> None:
    # Use the duplicates.json which has 3 items, 2 are identical
    curator = Curator(base_manifest)

    # Run
    final_data = curator.prepare_dataset()

    # Expect 2 items (3 original - 1 duplicate = 2)
    assert len(final_data) == 2
    # Verify structure
    assert "instruction" in final_data[0]
    assert "input" in final_data[0]
    assert "output" in final_data[0]


def test_curator_full_flow_orpo(base_manifest: TrainingManifest) -> None:
    base_manifest.method_config.type = MethodType.ORPO
    base_manifest.dataset.ref = "synthesis://tests/data/orpo_duplicates.json"

    curator = Curator(base_manifest)
    final_data = curator.prepare_dataset()

    assert len(final_data) == 2
    assert "prompt" in final_data[0]
    assert "chosen" in final_data[0]
    assert "rejected" in final_data[0]


def test_curator_skips_dedup(base_manifest: TrainingManifest) -> None:
    base_manifest.dataset.sem_dedup = False
    curator = Curator(base_manifest)
    final_data = curator.prepare_dataset()

    # Should keep all 3
    assert len(final_data) == 3
