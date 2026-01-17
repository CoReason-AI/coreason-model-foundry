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
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from coreason_model_foundry.schemas import (
    ComputeConfig,
    DatasetConfig,
    MethodConfig,
    MethodType,
    TrainingManifest,
)
from coreason_model_foundry.strategies import (
    DoRAStrategy,
    ORPOStrategy,
    QLoRAStrategy,
    StrategyFactory,
    TrainingStrategy,
)
# Import orpo module explicitly to patch it directly
from coreason_model_foundry.strategies import orpo


@pytest.fixture(autouse=True)
def global_mocks() -> Generator[None, None, None]:
    """Ensure heavy dependencies are mocked for all tests in this file."""
    # Create a mock torch module with a real class for Tensor so isinstance checks pass
    mock_torch = MagicMock()
    class MockTensor:
        pass
    mock_torch.Tensor = MockTensor
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.get_device_properties.return_value.total_memory = 25 * (1024**3)

    with patch.dict(
        sys.modules,
        {
            "unsloth": MagicMock(),
            "trl": MagicMock(),
            "datasets": MagicMock(),
            "transformers": MagicMock(),
            "torch": mock_torch,
        },
    ):
        # Specifically ensure Unsloth mock has the required attribute for DoRA/ORPO
        sys.modules["unsloth"].FastLanguageModel = MagicMock()
        sys.modules["unsloth"].PatchDPOTrainer = MagicMock()
        yield


@pytest.fixture
def base_manifest() -> TrainingManifest:
    return TrainingManifest(
        job_id="test-job-1",
        base_model="test-model",
        method_config=MethodConfig(
            type=MethodType.QLORA,
            rank=16,
            alpha=32,
            target_modules=["q_proj"],
            strict_hardware_check=False
        ),
        dataset=DatasetConfig(ref="test-dataset", dedup_threshold=0.95),
        compute=ComputeConfig(batch_size=1, grad_accum=1, context_window=1024, quantization="4bit"),
    )


def test_factory_creates_qlora(base_manifest: TrainingManifest) -> None:
    base_manifest.method_config.type = MethodType.QLORA
    strategy = StrategyFactory.get_strategy(base_manifest)
    assert isinstance(strategy, QLoRAStrategy)
    assert isinstance(strategy, TrainingStrategy)
    assert strategy.manifest == base_manifest


def test_factory_creates_dora(base_manifest: TrainingManifest) -> None:
    base_manifest.method_config.type = MethodType.DORA

    # We patch validate or the dependency.
    # Since we use global_mocks, unsloth is in sys.modules, but DoRA might have cached it as None.
    with patch("coreason_model_foundry.strategies.dora.FastLanguageModel", MagicMock()):
        strategy = StrategyFactory.get_strategy(base_manifest)
        assert isinstance(strategy, DoRAStrategy)


def test_factory_creates_orpo(base_manifest: TrainingManifest) -> None:
    base_manifest.method_config.type = MethodType.ORPO

    # Use patch.object on the imported module object to be robust against path variations
    with patch.object(orpo, "FastLanguageModel", MagicMock()), \
         patch.object(orpo, "torch", sys.modules["torch"]):

        strategy = StrategyFactory.get_strategy(base_manifest)
        assert isinstance(strategy, ORPOStrategy)


def test_factory_validation_error_unknown_type(base_manifest: TrainingManifest) -> None:
    # Mock the manifest to return an invalid type that isn't in the registry
    mock_manifest = MagicMock()
    mock_manifest.method_config.type = "INVALID_TYPE"

    with pytest.raises(ValueError, match="Unsupported training method"):
        StrategyFactory.get_strategy(mock_manifest)


def test_strategies_have_validate_and_train_methods(base_manifest: TrainingManifest) -> None:
    strategy = StrategyFactory.get_strategy(base_manifest)
    assert hasattr(strategy, "validate")
    assert hasattr(strategy, "train")

    # Test basic return of train (mocked) for QLoRA
    result = strategy.train([{}])  # Pass empty list of dicts
    assert result["status"] == "mock_success"


def test_dora_train_method(base_manifest: TrainingManifest) -> None:
    base_manifest.method_config.type = MethodType.DORA

    with (
        patch("coreason_model_foundry.strategies.dora.FastLanguageModel") as mock_flm,
        patch("coreason_model_foundry.strategies.dora.SFTTrainer"),
        patch("coreason_model_foundry.strategies.dora.Dataset") as mock_dataset,
        patch("coreason_model_foundry.strategies.dora.DoRAStrategy.validate"), # Bypass validate if needed
    ):
        mock_flm.from_pretrained.return_value = (MagicMock(), MagicMock())
        mock_flm.get_peft_model.return_value = MagicMock()
        mock_dataset.from_list.return_value = MagicMock()

        strategy = StrategyFactory.get_strategy(base_manifest)
        result = strategy.train([{"instruction": "i", "input": "i", "output": "o"}])
        assert result["status"] == "success"
        assert result["strategy"] == "dora"


def test_orpo_train_method(base_manifest: TrainingManifest) -> None:
    base_manifest.method_config.type = MethodType.ORPO

    with (
        patch.object(orpo, "FastLanguageModel") as mock_flm,
        patch.object(orpo, "ORPOTrainer") as mock_trainer,
        patch.object(orpo, "Dataset") as mock_dataset,
        patch.object(orpo, "PatchDPOTrainer"),
        patch.object(orpo, "torch", sys.modules["torch"]),
    ):
        mock_flm.from_pretrained.return_value = (MagicMock(), MagicMock())
        mock_flm.get_peft_model.return_value = MagicMock()
        mock_dataset.from_list.return_value = MagicMock()
        mock_trainer.return_value.train.return_value = None

        strategy = StrategyFactory.get_strategy(base_manifest)

        # Pass required triplet keys for ORPO
        result = strategy.train([{"prompt": "p", "chosen": "c", "rejected": "r"}])
        assert result["status"] == "success"
        assert result["strategy"] == "orpo"


def test_qlora_validation_warning_captured(base_manifest: TrainingManifest) -> None:
    """Test that QLoRA validation correctly logs a warning for non-4bit quantization."""
    base_manifest.method_config.type = MethodType.QLORA
    base_manifest.compute.quantization = "8bit"  # Not 4bit

    # Now that QLoRA imports logger, we can patch it
    with patch("coreason_model_foundry.strategies.qlora.logger") as mock_logger:
        StrategyFactory.get_strategy(base_manifest)
        mock_logger.warning.assert_called_once()
        assert "QLoRA usually requires 4bit quantization" in mock_logger.warning.call_args[0][0]


def test_validate_is_called_during_factory_creation(base_manifest: TrainingManifest) -> None:
    """Verify that the factory calls .validate() on the strategy."""
    base_manifest.method_config.type = MethodType.QLORA

    # We patch the QLoRAStrategy class specifically to spy on its `validate` method
    with patch.object(QLoRAStrategy, "validate", autospec=True) as mock_validate:
        strategy = StrategyFactory.get_strategy(base_manifest)
        mock_validate.assert_called_once_with(strategy)


def test_strategy_isolation(base_manifest: TrainingManifest) -> None:
    """Verify that strategies created by the factory are independent instances."""
    base_manifest.method_config.type = MethodType.QLORA

    strategy_1 = StrategyFactory.get_strategy(base_manifest)
    strategy_2 = StrategyFactory.get_strategy(base_manifest)

    assert strategy_1 is not strategy_2
    assert strategy_1.manifest is strategy_2.manifest  # Same manifest object passed in

    # Modify manifest for second strategy
    # If we create a new manifest, they should be different
    manifest_2 = base_manifest.model_copy(deep=True)
    manifest_2.job_id = "job-2"
    strategy_3 = StrategyFactory.get_strategy(manifest_2)

    assert strategy_3.manifest.job_id == "job-2"
    assert strategy_1.manifest.job_id == "test-job-1"
