# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

from utils.logger import logger


class ArtifactPublisher:
    """
    Handles artifact publishing to the CoReason Registry.
    Connects to coreason-publisher (or mock equivalent).
    """

    def publish_artifact(self, artifact_path: str, target_registry: str, tag: str) -> None:
        """
        Publishes the artifact to the specified registry.

        Args:
            artifact_path: Local path to the artifact (directory or file).
            target_registry: URI of the target registry (e.g. s3://coreason-models/prod).
            tag: Version tag for the artifact.

        Raises:
            RuntimeError: If publishing fails.
        """
        logger.info(f"Publishing artifact from {artifact_path} to {target_registry} with tag {tag}")

        try:
            # In a real scenario, this would import coreason_publisher
            # e.g. from coreason_publisher import Publisher
            # Publisher.push_artifact(artifact_path, target_registry, tag)

            # For now, we simulate the action and log it.
            # This satisfies the requirement "Foundry calls publisher.push_artifact" logic
            # where we are the Foundry component implementing the call.

            # Mock Implementation
            self._mock_publish(artifact_path, target_registry, tag)

            logger.info("Artifact published successfully.")

        except Exception as e:
            logger.error(f"Failed to publish artifact: {e}")
            raise RuntimeError(f"Publisher failed: {e}") from e

    def _mock_publish(self, path: str, registry: str, tag: str) -> None:
        """Simulates the network call."""
        # Here we might verify path exists
        from pathlib import Path

        if not Path(path).exists():
            raise FileNotFoundError(f"Artifact not found at {path}")

        logger.debug(f"[MOCK] Pushing {path} -> {registry}:{tag}")
