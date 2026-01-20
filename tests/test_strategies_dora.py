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
        publish_target=None,
        job_id="test-job-001",
        base_model="meta-llama/Meta-Llama-3-8B",
        method_config=MethodConfig(
            type=MethodType.DORA,
            rank=32,
            alpha=16,
            target_modules=["q_proj", "v_proj"],
            strict_hardware_check=False,
        ),
        dataset=DatasetConfig(sem_dedup=False, ref="synthesis://test_data", dedup_threshold=0.95),
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


def test_dora_formatting_logic(dora_manifest: TrainingManifest) -> None:
    """Test the internal formatting function robustly handles edge cases."""
    from coreason_model_foundry.strategies.dora import DoRAStrategy

    mock_tokenizer = MagicMock()
    mock_tokenizer.eos_token = "<|eos|>"

    # Capture the formatting function
    captured_formatting_func = None

    def capture_trainer(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal captured_formatting_func
        captured_formatting_func = kwargs.get("formatting_func")
        return MagicMock()

    with (
        patch("coreason_model_foundry.strategies.dora.FastLanguageModel") as MockFLM,
        patch("coreason_model_foundry.strategies.dora.SFTTrainer", side_effect=capture_trainer),
        patch("coreason_model_foundry.strategies.dora.Dataset"),
    ):
        MockFLM.from_pretrained.return_value = (MagicMock(), mock_tokenizer)

        strategy = DoRAStrategy(dora_manifest)
        strategy.validate()
        # Pass dummy data, we only care about extracting the function
        strategy.train([{"instruction": "a", "input": "b", "output": "c"}])

    assert captured_formatting_func is not None

    # Test Case 1: Standard Input
    batch = {"instruction": ["Do this"], "input": ["With this"], "output": ["Result"]}
    result = captured_formatting_func(batch)
    assert "### Instruction:\nDo this" in result[0]
    assert "### Input:\nWith this" in result[0]
    assert "### Response:\nResult" in result[0]
    assert result[0].endswith("<|eos|>")

    # Test Case 2: None Values (Edge Case)
    batch_none = {"instruction": [None], "input": [None], "output": [None]}
    result_none = captured_formatting_func(batch_none)
    # Should not crash and replace None with empty string
    assert "### Instruction:\n\n" in result_none[0]
    assert "### Input:\n\n" in result_none[0]


def test_dora_bf16_enabled(dora_manifest: TrainingManifest) -> None:
    """Test that BF16 is enabled when supported by hardware (mocked)."""
    from coreason_model_foundry.strategies.dora import DoRAStrategy

    # Mock torch.cuda.is_available and is_bf16_supported
    # We need to mock 'torch' inside 'dora.py' scope specifically for the helper function
    # The helper `is_bfloat16_supported` imports torch inside.
    # So we patch sys.modules["torch"] which is already done by global fixture,
    # but we need to configure it to return True for bf16.

    # Access the global mock from sys.modules
    mock_torch = sys.modules["torch"]
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.is_bf16_supported.return_value = True

    captured_args = None

    def capture_trainer(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal captured_args
        captured_args = kwargs.get("args")
        return MagicMock()

    with (
        patch("coreason_model_foundry.strategies.dora.FastLanguageModel") as MockFLM,
        patch("coreason_model_foundry.strategies.dora.SFTTrainer", side_effect=capture_trainer),
        patch("coreason_model_foundry.strategies.dora.Dataset"),
    ):
        MockFLM.from_pretrained.return_value = (MagicMock(), MagicMock())
        MockFLM.get_peft_model.return_value = MagicMock()

        strategy = DoRAStrategy(dora_manifest)
        strategy.validate()
        strategy.train([{"instruction": "a", "input": "b", "output": "c"}])

    assert captured_args is not None
    assert captured_args.bf16 is True
    assert captured_args.fp16 is False


def test_dora_train_missing_unsloth_runtime_error(
    dora_manifest: TrainingManifest, sample_dataset: List[Dict[str, str]]
) -> None:
    """Test that runtime error is raised if Unsloth is missing during train execution."""
    from coreason_model_foundry.strategies.dora import DoRAStrategy

    with patch("coreason_model_foundry.strategies.dora.FastLanguageModel", None):
        strategy = DoRAStrategy(dora_manifest)
        # We manually bypass validate to trigger error in train
        # or we assume validate passes but something happened to unsloth module
        # This covers the defensive check inside `train()`
        with pytest.raises(RuntimeError, match="Unsloth is required"):
            strategy.train(sample_dataset)


def test_dora_bf16_exception_handling(dora_manifest: TrainingManifest) -> None:
    """Test that exception handling in is_bfloat16_supported works."""
    from coreason_model_foundry.strategies.dora import DoRAStrategy

    # Access global mock
    mock_torch = sys.modules["torch"]
    # Force exception when checking cuda availability
    mock_torch.cuda.is_available.side_effect = Exception("CUDA Error")

    captured_args = None

    def capture_trainer(*args: Any, **kwargs: Any) -> MagicMock:
        nonlocal captured_args
        captured_args = kwargs.get("args")
        return MagicMock()

    with (
        patch("coreason_model_foundry.strategies.dora.FastLanguageModel") as MockFLM,
        patch("coreason_model_foundry.strategies.dora.SFTTrainer", side_effect=capture_trainer),
        patch("coreason_model_foundry.strategies.dora.Dataset"),
    ):
        MockFLM.from_pretrained.return_value = (MagicMock(), MagicMock())
        MockFLM.get_peft_model.return_value = MagicMock()

        strategy = DoRAStrategy(dora_manifest)
        strategy.validate()
        strategy.train([{"instruction": "a", "input": "b", "output": "c"}])

    # Reset side effect for other tests
    mock_torch.cuda.is_available.side_effect = None

    assert captured_args is not None
    assert captured_args.bf16 is False
    assert captured_args.fp16 is True  # Assuming is_bfloat16_supported returned False


def test_dora_module_level_import_error() -> None:
    """Test that missing unsloth at module level triggers warning and sets FLM to None."""
    # Temporarily remove unsloth and dora from sys.modules to simulate fresh import failure
    with patch.dict(sys.modules):
        # Remove unsloth to trigger import error (assuming it's not installed in env)
        # Or enforce it to raise ImportError if we can manipulate finders,
        # but simpler is to ensure it's not in sys.modules, and since it's not installed, it fails.
        if "unsloth" in sys.modules:
            del sys.modules["unsloth"]

        # Remove dora module so it re-executes top-level code
        if "coreason_model_foundry.strategies.dora" in sys.modules:
            del sys.modules["coreason_model_foundry.strategies.dora"]

        # We also need to patch logger to verify warning
        with patch("utils.logger.logger") as mock_logger:
            # We must import using importlib to force reload in this context?
            # Standard import should work since we deleted it from sys.modules
            import coreason_model_foundry.strategies.dora as dora_module

            # Since "unsloth" is missing, the try/except block runs
            assert dora_module.FastLanguageModel is None  # type: ignore[attr-defined]

            # Verify warning was logged
            # Note: logger might be initialized at module level of utils.logger
            # If we patched utils.logger.logger, it should be captured.
            # However, dora.py does `from utils.logger import logger`
            # If utils.logger is already imported, it gets the object.
            # Our patch `patch("utils.logger.logger")` patches the attribute `logger` in `utils.logger` module.
            # So `dora.py` imports that patched object.
            # BUT, we need to ensure `dora.py` re-imports `logger` or uses the patched one.
            # Yes, re-importing `dora` will run `from utils.logger import logger`.

            # Actually, `utils.logger` is likely already imported.
            # `patch` modifies the module attribute in place.
            # So `from utils.logger import logger` gets the Mock.

            mock_logger.warning.assert_called_with(
                "Unsloth not found. DoRA training will fail if executed on this environment."
            )
