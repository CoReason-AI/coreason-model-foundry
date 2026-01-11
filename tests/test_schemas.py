# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

from typing import Any, Dict

import pytest
from pydantic import ValidationError

from coreason_model_foundry.schemas import (
    MergeMethod,
    MergeRecipe,
    MethodType,
    TrainingManifest,
)


# Valid Data Fixtures
@pytest.fixture
def valid_method_config() -> Dict[str, Any]:
    return {"type": "dora", "rank": 64, "alpha": 16, "target_modules": ["q_proj", "v_proj"]}


@pytest.fixture
def valid_dataset_config() -> Dict[str, Any]:
    return {"ref": "synthesis://batch_1", "sem_dedup": True, "dedup_threshold": 0.95}


@pytest.fixture
def valid_compute_config() -> Dict[str, Any]:
    return {"batch_size": 4, "grad_accum": 4, "context_window": 8192, "quantization": "4bit"}


@pytest.fixture
def valid_manifest_data(
    valid_method_config: Dict[str, Any], valid_dataset_config: Dict[str, Any], valid_compute_config: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "job_id": "test-job-1",
        "base_model": "llama-3",
        "method_config": valid_method_config,
        "dataset": valid_dataset_config,
        "compute": valid_compute_config,
    }


@pytest.fixture
def valid_merge_recipe_data() -> Dict[str, Any]:
    return {
        "job_id": "merge-job-1",
        "merge_method": "dare_ties",
        "base_model": "llama-3",
        "models": [{"model": "adapter_1", "parameters": {"weight": 1.0, "density": 0.5}}],
        "dtype": "bfloat16",
    }


# Test TrainingManifest
def test_valid_training_manifest(valid_manifest_data: Dict[str, Any]) -> None:
    manifest = TrainingManifest(**valid_manifest_data)
    assert manifest.job_id == "test-job-1"
    assert manifest.method_config.type == MethodType.DORA


def test_invalid_method_type(valid_manifest_data: Dict[str, Any]) -> None:
    valid_manifest_data["method_config"]["type"] = "invalid_type"
    with pytest.raises(ValidationError) as excinfo:
        TrainingManifest(**valid_manifest_data)
    assert "Input should be 'qlora', 'dora' or 'orpo'" in str(excinfo.value)


def test_invalid_rank(valid_manifest_data: Dict[str, Any]) -> None:
    valid_manifest_data["method_config"]["rank"] = 0
    with pytest.raises(ValidationError):
        TrainingManifest(**valid_manifest_data)


def test_invalid_dedup_threshold(valid_manifest_data: Dict[str, Any]) -> None:
    valid_manifest_data["dataset"]["dedup_threshold"] = 1.5
    with pytest.raises(ValidationError):
        TrainingManifest(**valid_manifest_data)


def test_invalid_quantization(valid_manifest_data: Dict[str, Any]) -> None:
    valid_manifest_data["compute"]["quantization"] = "16bit"  # Not in allowed literal
    with pytest.raises(ValidationError):
        TrainingManifest(**valid_manifest_data)


# Test MergeRecipe
def test_valid_merge_recipe(valid_merge_recipe_data: Dict[str, Any]) -> None:
    recipe = MergeRecipe(**valid_merge_recipe_data)
    assert recipe.merge_method == MergeMethod.DARE_TIES
    assert len(recipe.models) == 1


def test_invalid_density(valid_merge_recipe_data: Dict[str, Any]) -> None:
    valid_merge_recipe_data["models"][0]["parameters"]["density"] = 1.1
    with pytest.raises(ValidationError):
        MergeRecipe(**valid_merge_recipe_data)


def test_invalid_dtype(valid_merge_recipe_data: Dict[str, Any]) -> None:
    valid_merge_recipe_data["dtype"] = "int8"
    with pytest.raises(ValidationError):
        MergeRecipe(**valid_merge_recipe_data)
