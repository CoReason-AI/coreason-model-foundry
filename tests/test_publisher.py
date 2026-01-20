# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from coreason_model_foundry.publisher import ArtifactPublisher


def test_publish_artifact_success(tmp_path: Path) -> None:
    artifact_path = tmp_path / "model"
    artifact_path.mkdir()
    registry = "s3://models"
    tag = "v1"

    publisher = ArtifactPublisher()

    # Since we are mocking the internal logic (which currently is just logging and a check),
    # we can verify the check works.

    publisher.publish_artifact(str(artifact_path), registry, tag)
    # If no exception, success.


def test_publish_artifact_not_found() -> None:
    publisher = ArtifactPublisher()
    with pytest.raises(RuntimeError) as exc:
        publisher.publish_artifact("non_existent_path", "s3://models", "v1")

    assert "Publisher failed" in str(exc.value)
    assert "Artifact not found" in str(exc.value.__cause__)


@patch("coreason_model_foundry.publisher.logger")
def test_publish_artifact_logging(mock_logger: MagicMock, tmp_path: Path) -> None:
    artifact_path = tmp_path / "model"
    artifact_path.mkdir()

    publisher = ArtifactPublisher()
    publisher.publish_artifact(str(artifact_path), "reg", "tag")

    mock_logger.info.assert_any_call(f"Publishing artifact from {artifact_path} to reg with tag tag")
    mock_logger.info.assert_any_call("Artifact published successfully.")
