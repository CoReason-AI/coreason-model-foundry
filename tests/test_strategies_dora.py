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
    # Create a mock torch module with a real class for Tensor so isinstance checks pass
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
            "torch": mock_torch,
        },
    ):
        # Mock specifically the Safe Import check in dora.py
        sys.modules["unsloth"].FastLanguageModel = MagicMock()  # type: ignore

        yield


@pytest.fixture
def dora_manifest() -> TrainingManifest:
    return TrainingManifest(
        job_id="test-job-001",
        base_model="meta-llama/Meta-Llama-3-8B",
        method_config=MethodConfig(type=MethodType.DORA, rank=32, alpha=16, target_modules=["q_proj", "v_proj"]),
        dataset=DatasetConfig(ref="synthesis://test_data", dedup_threshold=0.95),
        compute=ComputeConfig(batch_size=2, grad_accum=1, context_window=1024, quantization="4bit"),
    )


@pytest.fixture
def sample_dataset() -> List[Dict[str, str]]:
    return [
        {"instruction": "Add", "input": "1+1", "output": "2"},
        {"instruction": "Sub", "input": "2-1", "output": "1"},
    ]


def test_dora_train_success(dora_manifest: TrainingManifest, sample_dataset: List[Dict[str, str]]) -> None:
    """
    Test that DoRA strategy correctly initializes Unsloth and SFTTrainer
    and calls train.
    """
    # Import inside the test function to ensure mocks are active
    from coreason_model_foundry.strategies.dora import DoRAStrategy
    # from trl import SFTTrainer # Don't import real trl

    # Setup Mocks
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token = "<|end_of_text|>"

    mock_trainer = MagicMock()

    # We patch imports used inside dora.py
    with (
        patch("coreason_model_foundry.strategies.dora.FastLanguageModel") as MockFLM,
        patch("coreason_model_foundry.strategies.dora.SFTTrainer", return_value=mock_trainer) as MockSFTTrainer,
        patch("coreason_model_foundry.strategies.dora.Dataset") as MockDataset,
    ):
        MockFLM.from_pretrained.return_value = (mock_model, mock_tokenizer)
        MockFLM.get_peft_model.return_value = mock_model

        # Execution
        strategy = DoRAStrategy(dora_manifest)
        strategy.validate()
        result = strategy.train(sample_dataset)

        # Verification
        assert result["status"] == "success"
        assert result["job_id"] == "test-job-001"

        # 1. Verify Model Loading
        MockFLM.from_pretrained.assert_called_once_with(
            model_name="meta-llama/Meta-Llama-3-8B", max_seq_length=1024, dtype=None, load_in_4bit=True
        )

        # 2. Verify DoRA Config
        MockFLM.get_peft_model.assert_called_once()
        _, kwargs = MockFLM.get_peft_model.call_args
        assert kwargs["r"] == 32
        assert kwargs["target_modules"] == ["q_proj", "v_proj"]
        assert kwargs["use_dora"] is True  # Critical check

        # 3. Verify Dataset Conversion (Mocked)
        MockDataset.from_list.assert_called_once_with(sample_dataset)

        # 4. Verify Trainer Setup
        MockSFTTrainer.assert_called_once()
        call_kwargs = MockSFTTrainer.call_args[1]
        assert call_kwargs["model"] == mock_model
        assert call_kwargs["tokenizer"] == mock_tokenizer
        # Check formatting func
        assert "formatting_func" in call_kwargs

        # 5. Verify Training Execution
        mock_trainer.train.assert_called_once()
        mock_model.save_pretrained.assert_called_once()


def test_dora_train_empty_dataset(dora_manifest: TrainingManifest) -> None:
    from coreason_model_foundry.strategies.dora import DoRAStrategy

    # Patch FLM so validation passes
    with patch("coreason_model_foundry.strategies.dora.FastLanguageModel"):
        strategy = DoRAStrategy(dora_manifest)
        with pytest.raises(ValueError, match="Training dataset is empty"):
            strategy.train([])


def test_dora_missing_unsloth(dora_manifest: TrainingManifest) -> None:
    # Simulate unsloth missing
    # We explicitly patch FastLanguageModel to None in dora.py
    with patch("coreason_model_foundry.strategies.dora.FastLanguageModel", None):
        from coreason_model_foundry.strategies.dora import DoRAStrategy

        strategy = DoRAStrategy(dora_manifest)

        with pytest.raises(RuntimeError, match="Unsloth is required"):
            strategy.validate()
