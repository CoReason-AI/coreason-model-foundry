# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

import builtins
import importlib
import sys
from typing import Any, Generator, Optional
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


def test_orpo_cuda_query_exception() -> None:
    """Test that a generic CUDA error during validation is caught and logged, not raised."""
    mock_torch = sys.modules["torch"]
    mock_torch.cuda.get_device_properties.side_effect = RuntimeError("Unknown CUDA error")

    # Manifest
    manifest = TrainingManifest(
        job_id="test-job-orpo-cuda-err",
        base_model="meta-llama/Meta-Llama-3-8B",
        method_config=MethodConfig(
            type=MethodType.ORPO,
            rank=64,
            alpha=32,
            target_modules=["q_proj"],
            strict_hardware_check=True,
        ),
        dataset=DatasetConfig(ref="synthesis://test", dedup_threshold=0.95),
        compute=ComputeConfig(batch_size=1, grad_accum=1, context_window=1024),
    )

    with (
        patch("coreason_model_foundry.strategies.orpo.FastLanguageModel", MagicMock()),
        patch("coreason_model_foundry.strategies.orpo.torch", mock_torch),
        patch("coreason_model_foundry.strategies.orpo.logger") as mock_logger,
    ):
        from coreason_model_foundry.strategies.orpo import ORPOStrategy

        strategy = ORPOStrategy(manifest)

        # Should NOT raise exception, but log warning
        strategy.validate()

        # Verify warning logged
        mock_logger.warning.assert_called()
        assert "Could not query GPU properties" in mock_logger.warning.call_args[0][0]


def test_orpo_config_propagation() -> None:
    """Test that manifest configuration is accurately propagated to the model loader."""
    mock_torch = sys.modules["torch"]
    mock_torch.cuda.get_device_properties.return_value.total_memory = 32 * (1024**3)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    manifest = TrainingManifest(
        job_id="test-job-orpo-config",
        base_model="test-model-x",
        method_config=MethodConfig(
            type=MethodType.ORPO,
            rank=128,
            alpha=64,
            target_modules=target_modules,
            strict_hardware_check=True,
        ),
        dataset=DatasetConfig(ref="synthesis://test", dedup_threshold=0.95),
        compute=ComputeConfig(batch_size=4, grad_accum=2, context_window=4096, quantization="4bit"),
    )

    with (
        patch("coreason_model_foundry.strategies.orpo.FastLanguageModel") as MockFLM,
        patch("coreason_model_foundry.strategies.orpo.torch", mock_torch),
        patch("coreason_model_foundry.strategies.orpo.ORPOTrainer"),
        patch("coreason_model_foundry.strategies.orpo.Dataset"),
        patch("coreason_model_foundry.strategies.orpo.PatchDPOTrainer"),
    ):
        MockFLM.from_pretrained.return_value = (MagicMock(), MagicMock())
        MockFLM.get_peft_model.return_value = MagicMock()

        from coreason_model_foundry.strategies.orpo import ORPOStrategy

        strategy = ORPOStrategy(manifest)

        dataset = [{"prompt": "p", "chosen": "c", "rejected": "r"}]
        strategy.train(dataset)

        # Verify from_pretrained args
        MockFLM.from_pretrained.assert_called_once()
        _, kwargs = MockFLM.from_pretrained.call_args
        assert kwargs["model_name"] == "test-model-x"
        assert kwargs["max_seq_length"] == 4096
        assert kwargs["load_in_4bit"] is True

        # Verify get_peft_model args
        MockFLM.get_peft_model.assert_called_once()
        _, peft_kwargs = MockFLM.get_peft_model.call_args
        assert peft_kwargs["r"] == 128
        assert peft_kwargs["lora_alpha"] == 64
        assert peft_kwargs["target_modules"] == target_modules


def test_orpo_no_cuda_warning() -> None:
    """Test that a warning is logged if no CUDA device is available during validation."""
    mock_torch = sys.modules["torch"]
    mock_torch.cuda.is_available.return_value = False

    manifest = TrainingManifest(
        job_id="test-job-no-cuda",
        base_model="model",
        method_config=MethodConfig(
            type=MethodType.ORPO, rank=8, alpha=16, target_modules=["q"], strict_hardware_check=True
        ),
        dataset=DatasetConfig(ref="synthesis://test", dedup_threshold=0.95),
        compute=ComputeConfig(batch_size=1, grad_accum=1, context_window=1024),
    )

    with (
        patch("coreason_model_foundry.strategies.orpo.FastLanguageModel", MagicMock()),
        patch("coreason_model_foundry.strategies.orpo.torch", mock_torch),
        patch("coreason_model_foundry.strategies.orpo.logger") as mock_logger,
    ):
        from coreason_model_foundry.strategies.orpo import ORPOStrategy

        strategy = ORPOStrategy(manifest)
        strategy.validate()

        mock_logger.warning.assert_called_with(
            "No CUDA device detected. Validation skipped (assuming CPU/Mock environment)."
        )


def test_orpo_invalid_method_type() -> None:
    """Test validation raises ValueError if method type is not ORPO."""
    manifest = TrainingManifest(
        job_id="test-job-invalid",
        base_model="model",
        method_config=MethodConfig(
            type=MethodType.DORA,  # Invalid
            rank=8,
            alpha=16,
            target_modules=["q"],
            strict_hardware_check=True,
        ),
        dataset=DatasetConfig(ref="synthesis://test", dedup_threshold=0.95),
        compute=ComputeConfig(batch_size=1, grad_accum=1, context_window=1024),
    )

    with patch("coreason_model_foundry.strategies.orpo.FastLanguageModel", MagicMock()):
        from coreason_model_foundry.strategies.orpo import ORPOStrategy

        strategy = ORPOStrategy(manifest)
        with pytest.raises(ValueError, match="Invalid strategy type for ORPOStrategy"):
            strategy.validate()


def test_is_bfloat16_supported_exception() -> None:
    """Test helper function handles import or runtime errors."""
    from coreason_model_foundry.strategies.orpo import is_bfloat16_supported

    mock_torch = sys.modules["torch"]
    mock_torch.cuda.is_available.side_effect = Exception("Boom")

    assert is_bfloat16_supported() is False

    # Reset side effect
    mock_torch.cuda.is_available.side_effect = None


def test_orpo_import_error_unsloth() -> None:
    """Test that missing unsloth triggers a warning and sets FLM to None."""
    # We must patch sys.modules to remove unsloth, then force reload orpo.py

    # Create a clean environment without unsloth
    with patch.dict(sys.modules):
        if "unsloth" in sys.modules:
            del sys.modules["unsloth"]
        # Also ensure 'coreason_model_foundry.strategies.orpo' is reloaded
        if "coreason_model_foundry.strategies.orpo" in sys.modules:
            del sys.modules["coreason_model_foundry.strategies.orpo"]

        with patch("utils.logger.logger") as mock_logger:
            # Re-import
            import coreason_model_foundry.strategies.orpo as orpo_module

            importlib.reload(orpo_module)

            assert orpo_module.FastLanguageModel is None  # type: ignore
            assert orpo_module.PatchDPOTrainer is None  # type: ignore

            mock_logger.warning.assert_called_with(
                "Unsloth not found. ORPO training will fail if executed on this environment."
            )


def test_orpo_import_error_torch() -> None:
    """Test that missing torch sets torch to None."""

    real_import = builtins.__import__

    def side_effect(
        name: str,
        globals: Optional[dict[str, Any]] = None,
        locals: Optional[dict[str, Any]] = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> Any:
        if name == "torch":
            raise ImportError("Mocked ImportError for torch")
        return real_import(name, globals, locals, fromlist, level)

    # Remove torch from sys.modules to force re-import logic
    with patch.dict(sys.modules):
        # We don't delete torch from sys.modules because it causes segfault on reload
        # Instead, we just ensure orpo module is reloaded and the import inside it fails

        if "coreason_model_foundry.strategies.orpo" in sys.modules:
            del sys.modules["coreason_model_foundry.strategies.orpo"]

        with patch("builtins.__import__", side_effect=side_effect):
            # Re-import
            import coreason_model_foundry.strategies.orpo as orpo_module

            # Since import torch failed, orpo_module.torch should be None
            assert orpo_module.torch is None  # type: ignore
