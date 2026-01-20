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


# --- Mocks Setup ---
@pytest.fixture(autouse=True)
def mock_dependencies() -> Generator[None, None, None]:
    mock_torch = MagicMock()

    class MockTensor:
        pass

    mock_torch.Tensor = MockTensor

    with patch.dict(
        sys.modules,
        {
            "unsloth": MagicMock(),
            "trl": MagicMock(),
            "transformers": MagicMock(),
            "datasets": MagicMock(),
            # "torch": mock_torch, # We patch torch specifically in tests to handle local import
        },
    ):
        sys.modules["unsloth"].FastLanguageModel = MagicMock()  # type: ignore[attr-defined]
        yield


@pytest.fixture
def qlora_manifest() -> TrainingManifest:
    return TrainingManifest(
        job_id="test-job-qlora",
        base_model="meta-llama/Meta-Llama-3-8B",
        method_config=MethodConfig(
            type=MethodType.QLORA,
            rank=16,
            alpha=32,
            target_modules=["q_proj", "v_proj"],
            strict_hardware_check=True,
        ),
        dataset=DatasetConfig(ref="synthesis://test_data", dedup_threshold=0.95),
        compute=ComputeConfig(batch_size=2, grad_accum=1, context_window=1024, quantization="4bit"),
    )


@pytest.fixture
def sample_dataset() -> List[Dict[str, str]]:
    return [
        {"instruction": "Add", "input": "1+1", "output": "2"},
        {"instruction": "Sub", "input": "2-1", "output": "1"},
    ]


def test_qlora_training_flow(qlora_manifest: TrainingManifest, sample_dataset: List[Dict[str, str]]) -> None:
    """
    Test that QLoRA strategy correctly initializes Unsloth and SFTTrainer
    and calls train.
    """
    from coreason_model_foundry.strategies.qlora import QLoRAStrategy

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token = "<|end_of_text|>"
    mock_trainer = MagicMock()

    # Mock torch.cuda.is_available -> True
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    with (
        patch("coreason_model_foundry.strategies.qlora.FastLanguageModel") as MockFLM,
        patch("coreason_model_foundry.strategies.qlora.SFTTrainer", return_value=mock_trainer) as MockSFTTrainer,
        patch("coreason_model_foundry.strategies.qlora.Dataset"),
        patch("coreason_model_foundry.strategies.qlora.torch", mock_torch),
    ):
        MockFLM.from_pretrained.return_value = (mock_model, mock_tokenizer)
        MockFLM.get_peft_model.return_value = mock_model

        strategy = QLoRAStrategy(qlora_manifest)
        strategy.validate()
        result = strategy.train(sample_dataset)

        assert result["status"] == "success"
        assert result["job_id"] == "test-job-qlora"

        # 1. Verify Model Loading - MUST be 4bit
        MockFLM.from_pretrained.assert_called_once_with(
            model_name="meta-llama/Meta-Llama-3-8B", max_seq_length=1024, dtype=None, load_in_4bit=True
        )

        # 2. Verify PEFT Config
        MockFLM.get_peft_model.assert_called_once()
        _, kwargs = MockFLM.get_peft_model.call_args
        assert kwargs["r"] == 16
        assert kwargs["target_modules"] == ["q_proj", "v_proj"]

        # 3. Verify Trainer
        MockSFTTrainer.assert_called_once()
        mock_trainer.train.assert_called_once()
        mock_model.save_pretrained.assert_called_once()
        mock_tokenizer.save_pretrained.assert_called_once()


def test_qlora_no_gpu_failure(qlora_manifest: TrainingManifest) -> None:
    """
    Test that QLoRA fails if CUDA is not available.
    """
    from coreason_model_foundry.strategies.qlora import QLoRAStrategy

    # Mock torch.cuda.is_available -> False
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = False

    # We must patch FastLanguageModel so we don't fail on "Unsloth required" check
    # We must patch torch to simulate no GPU
    with (
        patch("coreason_model_foundry.strategies.qlora.FastLanguageModel"),
        patch("coreason_model_foundry.strategies.qlora.torch", mock_torch),
    ):
        strategy = QLoRAStrategy(qlora_manifest)

        with pytest.raises(EnvironmentError, match="QLoRA requires a GPU environment"):
            strategy.validate()


def test_qlora_quantization_override_warning(
    qlora_manifest: TrainingManifest, sample_dataset: List[Dict[str, str]]
) -> None:
    """
    Test that if quantization is not 4bit, it warns and overrides.
    """
    from coreason_model_foundry.strategies.qlora import QLoRAStrategy

    # Modify manifest to not be 4bit
    qlora_manifest.compute.quantization = "8bit"

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_trainer = MagicMock()

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    with (
        patch("coreason_model_foundry.strategies.qlora.FastLanguageModel") as MockFLM,
        patch("coreason_model_foundry.strategies.qlora.SFTTrainer", return_value=mock_trainer),
        patch("coreason_model_foundry.strategies.qlora.Dataset"),
        patch("coreason_model_foundry.strategies.qlora.logger") as mock_logger,
        patch("coreason_model_foundry.strategies.qlora.torch", mock_torch),
    ):
        MockFLM.from_pretrained.return_value = (mock_model, mock_tokenizer)
        MockFLM.get_peft_model.return_value = mock_model

        strategy = QLoRAStrategy(qlora_manifest)
        # Validation warns
        strategy.validate()
        mock_logger.warning.assert_any_call("QLoRA usually requires 4bit quantization.")

        # Train warns about override
        strategy.train(sample_dataset)
        # Check that we logged the override
        MockFLM.from_pretrained.assert_called_with(
            model_name="meta-llama/Meta-Llama-3-8B", max_seq_length=1024, dtype=None, load_in_4bit=True
        )


def test_qlora_missing_unsloth(qlora_manifest: TrainingManifest) -> None:
    """Test that missing unsloth raises RuntimeError during validation."""

    # Here we simulate FastLanguageModel being None
    with patch("coreason_model_foundry.strategies.qlora.FastLanguageModel", None):
        from coreason_model_foundry.strategies.qlora import QLoRAStrategy

        strategy = QLoRAStrategy(qlora_manifest)
        with pytest.raises(RuntimeError, match="Unsloth is required"):
            strategy.validate()
