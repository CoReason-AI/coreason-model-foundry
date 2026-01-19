# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

import sys
from typing import Dict, Generator, List
from unittest.mock import MagicMock, patch

import pytest

from coreason_model_foundry.schemas import (
    ComputeConfig,
    DatasetConfig,
    MethodConfig,
    MethodType,
    TrainingManifest,
)
from coreason_model_foundry.utils.hardware import HardwareIncompatibleError


# --- Mocks Setup ---
@pytest.fixture(autouse=True)
def mock_dependencies() -> Generator[None, None, None]:
    # Create a mock torch module with a real class for Tensor so isinstance checks pass
    mock_torch = MagicMock()

    class MockTensor:
        pass

    mock_torch.Tensor = MockTensor

    # Mock cuda properties
    mock_props = MagicMock()
    mock_props.total_memory = 24 * (1024**3)  # 24 GB by default
    mock_torch.cuda.get_device_properties.return_value = mock_props
    mock_torch.cuda.is_available.return_value = True

    with patch.dict(
        sys.modules,
        {
            "unsloth": MagicMock(),
            "trl": MagicMock(),
            "transformers": MagicMock(),
            "datasets": MagicMock(),
            "torch": mock_torch,
        },
    ):
        yield


@pytest.fixture
def orpo_manifest() -> TrainingManifest:
    return TrainingManifest(
        job_id="test-job-orpo-001",
        base_model="meta-llama/Meta-Llama-3-8B",
        method_config=MethodConfig(
            type=MethodType.ORPO, rank=64, alpha=32, target_modules=["q_proj", "v_proj"], strict_hardware_check=True
        ),
        dataset=DatasetConfig(ref="synthesis://test_orpo_data", dedup_threshold=0.95),
        compute=ComputeConfig(batch_size=2, grad_accum=1, context_window=2048, quantization="4bit"),
    )


@pytest.fixture
def sample_orpo_dataset() -> List[Dict[str, str]]:
    return [
        {"prompt": "Hi", "chosen": "Hello", "rejected": "Bye"},
        {"prompt": "Help", "chosen": "Sure", "rejected": "No"},
    ]


def test_orpo_validate_hardware_success(orpo_manifest: TrainingManifest) -> None:
    """Test validation passes when VRAM >= 24GB."""
    mock_torch = sys.modules["torch"]
    # Ensure it's exactly 24GB or more
    mock_torch.cuda.get_device_properties.return_value.total_memory = 25 * (1024**3)

    # Patch torch inside orpo module because it might be None if import failed earlier
    # Also patch hardware utils torch
    with (
        patch("coreason_model_foundry.strategies.orpo.FastLanguageModel", MagicMock()),
        patch("coreason_model_foundry.strategies.orpo.torch", mock_torch),
    ):
        from coreason_model_foundry.strategies.orpo import ORPOStrategy

        strategy = ORPOStrategy(orpo_manifest)
        # Should not raise (quantization is 4bit, so check skipped, but good to test)
        strategy.validate()


def test_orpo_validate_hardware_fail_fast_full_precision(orpo_manifest: TrainingManifest) -> None:
    """Test validation fails when VRAM < 24GB and quantization is 'none'."""
    mock_torch = sys.modules["torch"]
    # 16 GB
    mock_torch.cuda.get_device_properties.return_value.total_memory = 16 * (1024**3)

    orpo_manifest.compute.quantization = "none"

    with (
        patch("coreason_model_foundry.strategies.orpo.FastLanguageModel", MagicMock()),
        patch("coreason_model_foundry.strategies.orpo.torch", mock_torch),
    ):
        from coreason_model_foundry.strategies.orpo import ORPOStrategy

        strategy = ORPOStrategy(orpo_manifest)

        with pytest.raises(HardwareIncompatibleError, match="Insufficient VRAM"):
            strategy.validate()


def test_orpo_validate_hardware_skip_check_4bit(orpo_manifest: TrainingManifest) -> None:
    """Test validation skips check (or passes) for 4bit even with low VRAM."""
    mock_torch = sys.modules["torch"]
    mock_torch.cuda.get_device_properties.return_value.total_memory = 16 * (1024**3)

    orpo_manifest.compute.quantization = "4bit"

    with (
        patch("coreason_model_foundry.strategies.orpo.FastLanguageModel", MagicMock()),
        patch("coreason_model_foundry.strategies.orpo.torch", mock_torch),
    ):
        from coreason_model_foundry.strategies.orpo import ORPOStrategy

        strategy = ORPOStrategy(orpo_manifest)
        # Should NOT raise because we only check for quantization='none'
        strategy.validate()


def test_orpo_validate_hardware_8bit_pass(orpo_manifest: TrainingManifest) -> None:
    """Test validation passes for 8bit quantization even with low VRAM."""
    mock_torch = sys.modules["torch"]
    mock_torch.cuda.get_device_properties.return_value.total_memory = 16 * (1024**3)

    orpo_manifest.compute.quantization = "8bit"

    with (
        patch("coreason_model_foundry.strategies.orpo.FastLanguageModel", MagicMock()),
        patch("coreason_model_foundry.strategies.orpo.torch", mock_torch),
    ):
        from coreason_model_foundry.strategies.orpo import ORPOStrategy

        strategy = ORPOStrategy(orpo_manifest)
        # Should NOT raise
        strategy.validate()


def test_orpo_train_success(orpo_manifest: TrainingManifest, sample_orpo_dataset: List[Dict[str, str]]) -> None:
    """Test that ORPO strategy correctly initializes Unsloth and ORPOTrainer and calls train."""

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_trainer = MagicMock()

    # We patch the names in the module directly
    with (
        patch("coreason_model_foundry.strategies.orpo.FastLanguageModel") as MockFLM,
        patch("coreason_model_foundry.strategies.orpo.ORPOTrainer", return_value=mock_trainer) as MockORPOTrainer,
        patch("coreason_model_foundry.strategies.orpo.Dataset") as MockDataset,
        patch("coreason_model_foundry.strategies.orpo.PatchDPOTrainer") as MockPatch,
        patch("coreason_model_foundry.strategies.orpo.torch", sys.modules["torch"]),
    ):
        from coreason_model_foundry.strategies.orpo import ORPOStrategy

        MockFLM.from_pretrained.return_value = (mock_model, mock_tokenizer)
        MockFLM.get_peft_model.return_value = mock_model

        strategy = ORPOStrategy(orpo_manifest)
        strategy.validate()
        result = strategy.train(sample_orpo_dataset)

        assert result["status"] == "success"

        # Verify PatchDPOTrainer called
        MockPatch.assert_called_once()

        # Verify Model Loading
        MockFLM.from_pretrained.assert_called_once()

        # Verify LoRA setup
        MockFLM.get_peft_model.assert_called_once()
        _, kwargs = MockFLM.get_peft_model.call_args
        assert kwargs["target_modules"] == ["q_proj", "v_proj"]

        # Verify Dataset
        MockDataset.from_list.assert_called_once_with(sample_orpo_dataset)

        # Verify ORPO Trainer
        MockORPOTrainer.assert_called_once()
        call_kwargs = MockORPOTrainer.call_args[1]
        assert call_kwargs["model"] == mock_model
        assert call_kwargs["tokenizer"] == mock_tokenizer

        # Verify Train
        mock_trainer.train.assert_called_once()


def test_orpo_train_empty_dataset(orpo_manifest: TrainingManifest) -> None:
    with (
        patch("coreason_model_foundry.strategies.orpo.FastLanguageModel", MagicMock()),
        patch("coreason_model_foundry.strategies.orpo.torch", sys.modules["torch"]),
    ):
        from coreason_model_foundry.strategies.orpo import ORPOStrategy

        strategy = ORPOStrategy(orpo_manifest)
        with pytest.raises(ValueError, match="Training dataset is empty"):
            strategy.train([])
