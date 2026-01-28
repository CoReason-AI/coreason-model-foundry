# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import yaml

from coreason_identity.models import UserContext
from coreason_identity.types import SecretStr
from coreason_model_foundry.main import main
from coreason_model_foundry.schemas import TrainingManifest
from coreason_model_foundry.service import ModelFoundryServiceAsync

# Sample Manifest Data
MANIFEST_YAML = """
job_id: "test-job-001"
base_model: "meta-llama/Meta-Llama-3-8B"
method_config:
  type: "dora"
  rank: 64
  alpha: 16
  target_modules: ["q_proj", "v_proj"]
dataset:
  ref: "synthesis://test_dataset"
  sem_dedup: true
  dedup_threshold: 0.95
compute:
  batch_size: 4
  grad_accum: 4
  context_window: 8192
  quantization: "4bit"
"""


@pytest.fixture
def mock_manifest_file(tmp_path: Path) -> Path:
    p = tmp_path / "manifest.yaml"
    p.write_text(MANIFEST_YAML)
    return p


@pytest.fixture
def mock_context() -> UserContext:
    return UserContext(user_id=SecretStr("test-user"), roles=["tester"])


@pytest.mark.asyncio
async def test_load_manifest_valid(mock_manifest_file: Path) -> None:
    service = ModelFoundryServiceAsync()
    manifest = await service.load_manifest(mock_manifest_file)
    assert isinstance(manifest, TrainingManifest)
    assert manifest.job_id == "test-job-001"
    assert manifest.method_config.type == "dora"


@pytest.mark.asyncio
async def test_load_manifest_not_found() -> None:
    service = ModelFoundryServiceAsync()
    with pytest.raises(FileNotFoundError):
        await service.load_manifest(Path("non_existent.yaml"))


@pytest.mark.asyncio
async def test_load_manifest_invalid_yaml(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("job_id: [broken yaml")
    service = ModelFoundryServiceAsync()
    with pytest.raises(yaml.YAMLError):
        await service.load_manifest(p)


@pytest.mark.asyncio
async def test_calculate_provenance_id() -> None:
    manifest = MagicMock()
    manifest.model_dump_json.return_value = '{"job": "1"}'
    dataset = [{"data": "1"}]

    service = ModelFoundryServiceAsync()
    # Check consistency
    pid1 = await service.calculate_provenance_id(manifest, dataset)
    pid2 = await service.calculate_provenance_id(manifest, dataset)
    assert pid1 == pid2
    assert len(pid1) == 64  # SHA256 hex digest length


@pytest.mark.asyncio
async def test_calculate_provenance_id_sensitivity() -> None:
    """Complex Scenario: Verify hash sensitivity to changes."""
    manifest_a = MagicMock()
    manifest_a.model_dump_json.return_value = '{"job": "1"}'

    manifest_b = MagicMock()
    manifest_b.model_dump_json.return_value = '{"job": "2"}'

    dataset_a = [{"data": "1"}]
    dataset_b = [{"data": "2"}]

    service = ModelFoundryServiceAsync()
    pid_aa = await service.calculate_provenance_id(manifest_a, dataset_a)
    pid_ab = await service.calculate_provenance_id(manifest_a, dataset_b)
    pid_ba = await service.calculate_provenance_id(manifest_b, dataset_a)

    assert pid_aa != pid_ab, "Hash should change if dataset changes"
    assert pid_aa != pid_ba, "Hash should change if manifest changes"
    assert pid_ab != pid_ba


@patch("coreason_model_foundry.service.ArtifactPublisher")
@patch("coreason_model_foundry.service.Curator")
@patch("coreason_model_foundry.service.StrategyFactory")
@patch("coreason_model_foundry.service.ModelFoundryServiceAsync.load_manifest")
@pytest.mark.asyncio
async def test_orchestrate_training_flow(
    mock_load: AsyncMock,
    mock_factory: MagicMock,
    mock_curator_cls: MagicMock,
    mock_publisher_cls: MagicMock,
    mock_context: UserContext,
) -> None:
    # Setup Mocks
    mock_manifest = MagicMock()
    mock_manifest.job_id = "test-job"
    # Ensure nested attributes are mocked
    mock_manifest.method_config.type = "dora"
    # Ensure model_dump_json returns a string so encode works
    mock_manifest.model_dump_json.return_value = '{"job_id": "test-job"}'

    # Configure Publish Target
    mock_manifest.publish_target.registry = "s3://test"
    mock_manifest.publish_target.tag = "v1"

    mock_load.return_value = mock_manifest

    mock_curator_instance = mock_curator_cls.return_value
    mock_curator_instance.prepare_dataset.return_value = [{"instruction": "foo", "output": "bar"}]

    mock_strategy = MagicMock()
    mock_strategy.train.return_value = {
        "status": "success",
        "output_dir": "artifacts/test",
    }
    mock_factory.get_strategy.return_value = mock_strategy

    mock_publisher_instance = mock_publisher_cls.return_value

    # Execute
    async with ModelFoundryServiceAsync() as service:
        await service.orchestrate_training(Path("dummy_path.yaml"), context=mock_context)

    # Verify Interactions
    mock_load.assert_called_once_with(Path("dummy_path.yaml"))

    # Curator
    mock_curator_cls.assert_called_once_with(mock_manifest)
    # Since we wrap curator in to_thread, we don't mock async call but the method call
    mock_curator_instance.prepare_dataset.assert_called_once()

    # Strategy
    mock_factory.get_strategy.assert_called_once_with(mock_manifest)
    mock_strategy.train.assert_called_once()

    # Publisher
    mock_publisher_cls.assert_called_once()
    mock_publisher_instance.publish_artifact.assert_called_once_with(
        "artifacts/test", "s3://test", "v1", context=mock_context
    )


@patch("coreason_model_foundry.service.Curator")
@patch("coreason_model_foundry.service.ModelFoundryServiceAsync.load_manifest")
@pytest.mark.asyncio
async def test_orchestrate_training_empty_dataset(
    mock_load: AsyncMock, mock_curator_cls: MagicMock, mock_context: UserContext
) -> None:
    # Setup
    mock_load.return_value = MagicMock()
    mock_curator_instance = mock_curator_cls.return_value
    mock_curator_instance.prepare_dataset.return_value = []  # Empty dataset

    # Expect SystemExit(1)
    service = ModelFoundryServiceAsync()
    with pytest.raises(SystemExit) as exc:
        await service.orchestrate_training(Path("dummy.yaml"), context=mock_context)

    assert exc.value.code == 1
    await service._client.aclose()


@patch("coreason_model_foundry.service.ModelFoundryServiceAsync.load_manifest")
@pytest.mark.asyncio
async def test_orchestrate_training_exception(mock_load: AsyncMock, mock_context: UserContext) -> None:
    # Simulate an error during manifest loading
    mock_load.side_effect = ValueError("Critical Failure")

    service = ModelFoundryServiceAsync()
    with pytest.raises(ValueError, match="Critical Failure"):
        await service.orchestrate_training(Path("dummy.yaml"), context=mock_context)
    await service._client.aclose()


@patch("coreason_model_foundry.service.ModelFoundryService.orchestrate_training")
def test_main_cli_train(mock_orchestrate: MagicMock) -> None:
    test_args = ["main.py", "train", "--manifest", "config.yaml"]
    with patch.object(sys, "argv", test_args):
        main()

    # Assert that it was called with path and context (system context created inside main)
    mock_orchestrate.assert_called_once()
    args, kwargs = mock_orchestrate.call_args
    assert args[0] == Path("config.yaml")
    assert "context" in kwargs
    assert isinstance(kwargs["context"], UserContext)
    assert kwargs["context"].user_id.get_secret_value() == "cli-user"


@patch("coreason_model_foundry.service.ModelFoundryService.evaluate_model")
def test_main_cli_evaluate(mock_evaluate: MagicMock) -> None:
    test_args = ["main.py", "evaluate", "--model", "model_dir"]
    with patch.object(sys, "argv", test_args):
        main()

    mock_evaluate.assert_called_once()
    args, kwargs = mock_evaluate.call_args
    assert args[0] == Path("model_dir")
    assert "context" in kwargs
    assert isinstance(kwargs["context"], UserContext)


@patch("coreason_model_foundry.service.ModelFoundryService.publish_model")
def test_main_cli_publish(mock_publish: MagicMock) -> None:
    test_args = ["main.py", "publish", "--model", "model_dir", "--registry", "s3://reg", "--tag", "v1"]
    with patch.object(sys, "argv", test_args):
        main()

    mock_publish.assert_called_once()
    args, kwargs = mock_publish.call_args
    assert args[0] == Path("model_dir")
    assert args[1] == "s3://reg"
    assert args[2] == "v1"
    assert "context" in kwargs
    assert isinstance(kwargs["context"], UserContext)


def test_service_sync_facade(mock_manifest_file: Path, mock_context: UserContext) -> None:
    """Tests the sync facade execution using the real async logic (integration test-ish)."""
    # This might fail if it tries to actually do things, so we should probably mock internal parts
    # But for now, let's just check if it instantiates and calls orchestrated training.

    from coreason_model_foundry.service import ModelFoundryService

    with patch("coreason_model_foundry.service.ModelFoundryServiceAsync.orchestrate_training") as mock_async_orch:
        with ModelFoundryService() as service:
            # Also test the string path conversion
            service.orchestrate_training(str(mock_manifest_file), context=mock_context)

        mock_async_orch.assert_called_once_with(mock_manifest_file, context=mock_context)

    with patch("coreason_model_foundry.service.ModelFoundryServiceAsync.evaluate_model") as mock_async_eval:
        with ModelFoundryService() as service:
            service.evaluate_model("model_path", context=mock_context)
        mock_async_eval.assert_called_once_with(Path("model_path"), context=mock_context)

    with patch("coreason_model_foundry.service.ModelFoundryServiceAsync.publish_model") as mock_async_pub:
        with ModelFoundryService() as service:
            service.publish_model("model_path", "reg", "tag", context=mock_context)
        mock_async_pub.assert_called_once_with(Path("model_path"), "reg", "tag", context=mock_context)


@pytest.mark.asyncio
async def test_evaluate_model(mock_context: UserContext) -> None:
    service = ModelFoundryServiceAsync()
    with patch("coreason_model_foundry.service.logger") as mock_logger:
        await service.evaluate_model(Path("model"), context=mock_context)
        # Check logs
        assert mock_logger.info.call_count >= 2


@patch("coreason_model_foundry.service.ArtifactPublisher.publish_artifact")
@pytest.mark.asyncio
async def test_publish_model(mock_publish: MagicMock, mock_context: UserContext) -> None:
    service = ModelFoundryServiceAsync()
    await service.publish_model(Path("model"), "reg", "tag", context=mock_context)
    mock_publish.assert_called_once_with("model", "reg", "tag", context=mock_context)


# --- Coverage Filler Tests ---


@patch("coreason_model_foundry.service.ArtifactPublisher")
@patch("coreason_model_foundry.service.Curator")
@patch("coreason_model_foundry.service.StrategyFactory")
@patch("coreason_model_foundry.service.ModelFoundryServiceAsync.load_manifest")
@pytest.mark.asyncio
async def test_orchestrate_training_no_publish_target(
    mock_load: AsyncMock,
    mock_factory: MagicMock,
    mock_curator_cls: MagicMock,
    mock_publisher_cls: MagicMock,
    mock_context: UserContext,
) -> None:
    """Test when publish_target is None (default)."""
    # Setup
    mock_manifest = MagicMock()
    mock_manifest.job_id = "test-job-no-pub"
    mock_manifest.method_config.type = "dora"
    mock_manifest.model_dump_json.return_value = '{"job_id": "test-job"}'
    mock_manifest.publish_target = None  # EXPLICITLY NONE

    mock_load.return_value = mock_manifest

    # Execution should proceed without publishing
    mock_curator_instance = mock_curator_cls.return_value
    mock_curator_instance.prepare_dataset.return_value = [{"data": "val"}]

    mock_strategy = MagicMock()
    mock_strategy.train.return_value = {"output_dir": "artifacts/test"}
    mock_factory.get_strategy.return_value = mock_strategy

    # Act
    service = ModelFoundryServiceAsync()
    await service.orchestrate_training(Path("dummy.yaml"), context=mock_context)
    await service._client.aclose()

    # Assert
    mock_publisher_cls.assert_called_once()  # Instantiated in init
    mock_publisher_cls.return_value.publish_artifact.assert_not_called()


@patch("coreason_model_foundry.service.ArtifactPublisher")
@patch("coreason_model_foundry.service.Curator")
@patch("coreason_model_foundry.service.StrategyFactory")
@patch("coreason_model_foundry.service.ModelFoundryServiceAsync.load_manifest")
@patch("coreason_model_foundry.service.logger")
@pytest.mark.asyncio
async def test_orchestrate_training_publish_target_no_output_dir(
    mock_logger: MagicMock,
    mock_load: AsyncMock,
    mock_factory: MagicMock,
    mock_curator_cls: MagicMock,
    mock_publisher_cls: MagicMock,
    mock_context: UserContext,
) -> None:
    """Test when publish_target is set but strategy returns no output_dir."""
    # Setup
    mock_manifest = MagicMock()
    mock_manifest.job_id = "test-job-fail-pub"
    mock_manifest.method_config.type = "dora"
    mock_manifest.model_dump_json.return_value = '{"job_id": "test-job"}'

    # Configured publisher
    mock_manifest.publish_target.registry = "s3://test"
    mock_manifest.publish_target.tag = "v1"

    mock_load.return_value = mock_manifest

    # Strategy returns NO output_dir
    mock_strategy = MagicMock()
    mock_strategy.train.return_value = {"status": "success"}  # Missing output_dir
    mock_factory.get_strategy.return_value = mock_strategy

    mock_curator_instance = mock_curator_cls.return_value
    mock_curator_instance.prepare_dataset.return_value = [{"data": "val"}]

    # Act
    service = ModelFoundryServiceAsync()
    await service.orchestrate_training(Path("dummy.yaml"), context=mock_context)
    await service._client.aclose()

    # Assert
    mock_publisher_cls.return_value.publish_artifact.assert_not_called()
    mock_logger.warning.assert_any_call("No output directory returned from strategy. Skipping publication.")


def test_main_execution() -> None:
    """Verify that main.py can be executed (covers __name__ == '__main__' block indirectly)."""
    # This subprocess call runs the file, which triggers the if __name__ == "__main__": block
    result = subprocess.run(
        [sys.executable, "-m", "coreason_model_foundry.main", "--help"], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0
    assert "usage:" in result.stdout
