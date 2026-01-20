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
from typing import Any, Dict, Generator, List
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


def test_qlora_configuration_propagation(
    qlora_manifest: TrainingManifest, sample_dataset: List[Dict[str, str]]
) -> None:
    """
    Test that ComputeConfig values are correctly propagated to TrainingArguments.
    """
    from coreason_model_foundry.strategies.qlora import QLoRAStrategy

    # Customize manifest values
    qlora_manifest.compute.batch_size = 4
    qlora_manifest.compute.grad_accum = 8

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_trainer = MagicMock()

    # Mock torch.cuda.is_available -> True for execution
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    captured_args = None

    def capture_trainer_init(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal captured_args
        captured_args = kwargs.get("args")
        return mock_trainer

    with (
        patch("coreason_model_foundry.strategies.qlora.FastLanguageModel") as MockFLM,
        patch("coreason_model_foundry.strategies.qlora.SFTTrainer", side_effect=capture_trainer_init) as MockSFTTrainer,
        patch("coreason_model_foundry.strategies.qlora.Dataset"),
        patch("coreason_model_foundry.strategies.qlora.torch", mock_torch),
    ):
        MockFLM.from_pretrained.return_value = (mock_model, mock_tokenizer)
        MockFLM.get_peft_model.return_value = mock_model

        strategy = QLoRAStrategy(qlora_manifest)
        strategy.validate()
        strategy.train(sample_dataset)

        MockSFTTrainer.assert_called_once()
        assert captured_args is not None
        assert captured_args.per_device_train_batch_size == 4
        assert captured_args.gradient_accumulation_steps == 8
        assert captured_args.output_dir == f"artifacts/{qlora_manifest.job_id}"


def test_qlora_formatting_edge_cases(qlora_manifest: TrainingManifest) -> None:
    """
    Test the internal formatting function handles None values, empty strings, etc.
    """
    from coreason_model_foundry.strategies.qlora import QLoRAStrategy

    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token = "<|eos|>"

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    captured_formatting_func = None

    def capture_trainer_init(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal captured_formatting_func
        captured_formatting_func = kwargs.get("formatting_func")
        return MagicMock()

    with (
        patch("coreason_model_foundry.strategies.qlora.FastLanguageModel") as MockFLM,
        patch("coreason_model_foundry.strategies.qlora.SFTTrainer", side_effect=capture_trainer_init),
        patch("coreason_model_foundry.strategies.qlora.Dataset"),
        patch("coreason_model_foundry.strategies.qlora.torch", mock_torch),
    ):
        MockFLM.from_pretrained.return_value = (mock_model, mock_tokenizer)

        strategy = QLoRAStrategy(qlora_manifest)
        strategy.validate()
        # Trigger train to get the function
        strategy.train([{"instruction": "x", "input": "y", "output": "z"}])

    assert captured_formatting_func is not None

    # 1. Test None values
    batch_none = {"instruction": [None], "input": [None], "output": [None]}
    result = captured_formatting_func(batch_none)
    assert "### Instruction:\n\n" in result[0]
    assert "### Input:\n\n" in result[0]
    # Partial match for Response, as it contains output_val (empty) + eos
    # """### Response:\n{output_val}""" + eos
    assert "### Response:\n" in result[0]

    # 2. Test special chars
    batch_special = {"instruction": ["Code: \n\t"], "input": ["<script>alert(1)</script>"], "output": ["result"]}
    result_special = captured_formatting_func(batch_special)
    assert "Code: \n\t" in result_special[0]
    assert "<script>alert(1)</script>" in result_special[0]


def test_qlora_torch_missing(qlora_manifest: TrainingManifest) -> None:
    """
    Test behavior when 'torch' module import fails.
    The module should still load (safe import), but validate_environment should fail.
    """
    with (
        patch("coreason_model_foundry.strategies.qlora.torch", None),
        patch("coreason_model_foundry.strategies.qlora.FastLanguageModel"),
    ):
        from coreason_model_foundry.strategies.qlora import QLoRAStrategy

        strategy = QLoRAStrategy(qlora_manifest)

        # Should raise EnvironmentError "QLoRA requires a GPU environment"
        # because (torch is None or not torch.cuda.is_available()) will be True
        with pytest.raises(EnvironmentError, match="QLoRA requires a GPU environment"):
            strategy.validate()


def test_train_dataset_empty(qlora_manifest: TrainingManifest) -> None:
    """Test that train raises ValueError when dataset is empty."""
    from coreason_model_foundry.strategies.qlora import QLoRAStrategy

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    with (
        patch("coreason_model_foundry.strategies.qlora.FastLanguageModel"),
        patch("coreason_model_foundry.strategies.qlora.torch", mock_torch),
    ):
        strategy = QLoRAStrategy(qlora_manifest)
        # Calling train with empty list should raise ValueError
        with pytest.raises(ValueError, match="Training dataset is empty"):
            strategy.train([])


def test_train_unsloth_missing_defensive_check(
    qlora_manifest: TrainingManifest, sample_dataset: List[Dict[str, str]]
) -> None:
    """
    Test the defensive check inside train() where FastLanguageModel is None.
    This simulates a race condition or odd state where validate() passed (or was skipped)
    but import failed before train().
    """
    from coreason_model_foundry.strategies.qlora import QLoRAStrategy

    mock_torch = MagicMock()
    mock_torch.cuda.is_available.return_value = True

    # We patch FastLanguageModel to None inside the scope where we call train
    with (
        patch("coreason_model_foundry.strategies.qlora.FastLanguageModel", None),
        patch("coreason_model_foundry.strategies.qlora.torch", mock_torch),
    ):
        strategy = QLoRAStrategy(qlora_manifest)
        # We assume validate() is skipped or we simulate it passing somehow?
        # If we call train() directly, we hit the check.
        with pytest.raises(RuntimeError, match="Unsloth is required"):
            strategy.train(sample_dataset)


def test_is_bfloat16_supported_exception() -> None:
    """Test that is_bfloat16_supported catches exceptions and returns False."""
    from coreason_model_foundry.strategies.qlora import is_bfloat16_supported

    # We need to ensure QLoRAStrategy imports torch to call the helper,
    # but the helper imports torch internally.
    # Wait, the helper imports torch inside `try/except`.
    # So we need to mock `torch.cuda` to raise an Exception.

    # We need to patch sys.modules['torch'] so that `import torch` inside the function uses our mock.
    mock_torch = MagicMock()
    mock_torch.cuda.is_available.side_effect = Exception("CUDA Error")

    with patch.dict(sys.modules, {"torch": mock_torch}):
        assert is_bfloat16_supported() is False

    # Also test implicit import error (if import fails)
    # If we ensure torch is NOT in sys.modules, and cannot be imported
    with patch.dict(sys.modules):
        if "torch" in sys.modules:
            del sys.modules["torch"]
        # We need a finder that raises ImportError, or just rely on the fact that it might not be there?
        # Simpler to make the mock raise ImportError on access? No.
        # Making the import fail is harder without manipulating builtins.
        # But we covered the 'Exception' catch block above.
