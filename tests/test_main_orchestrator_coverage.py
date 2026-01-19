# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

from unittest.mock import MagicMock, patch

from coreason_model_foundry.main import orchestrate_training


@patch("coreason_model_foundry.main.ArtifactPublisher")
@patch("coreason_model_foundry.main.Curator")
@patch("coreason_model_foundry.main.StrategyFactory")
@patch("coreason_model_foundry.main.load_manifest")
def test_orchestrate_training_no_publish_target(
    mock_load: MagicMock,
    mock_factory: MagicMock,
    mock_curator_cls: MagicMock,
    mock_publisher_cls: MagicMock,
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
    orchestrate_training("dummy.yaml")

    # Assert
    mock_publisher_cls.assert_not_called()


@patch("coreason_model_foundry.main.ArtifactPublisher")
@patch("coreason_model_foundry.main.Curator")
@patch("coreason_model_foundry.main.StrategyFactory")
@patch("coreason_model_foundry.main.load_manifest")
@patch("coreason_model_foundry.main.logger")
def test_orchestrate_training_publish_target_no_output_dir(
    mock_logger: MagicMock,
    mock_load: MagicMock,
    mock_factory: MagicMock,
    mock_curator_cls: MagicMock,
    mock_publisher_cls: MagicMock,
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
    orchestrate_training("dummy.yaml")

    # Assert
    mock_publisher_cls.assert_not_called()
    mock_logger.warning.assert_any_call("No output directory returned from strategy. Skipping publication.")
