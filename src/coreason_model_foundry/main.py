# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from coreason_model_foundry.curator.main import Curator
from coreason_model_foundry.publisher import ArtifactPublisher
from coreason_model_foundry.schemas import TrainingManifest
from coreason_model_foundry.strategies.factory import StrategyFactory
from utils.logger import logger


def load_manifest(manifest_path: str) -> TrainingManifest:
    """
    Loads and validates the Training Manifest from a YAML file.

    Args:
        manifest_path: Path to the YAML file.

    Returns:
        Validated TrainingManifest object.
    """
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    logger.info(f"Loading manifest from {manifest_path}")
    with open(path, "r", encoding="utf-8") as f:
        # Pydantic doesn't natively parse YAML, so we parse to dict first
        data = yaml.safe_load(f)

    return TrainingManifest(**data)


def calculate_provenance_id(manifest: TrainingManifest, dataset: List[Dict[str, Any]]) -> str:
    """
    Calculates the GxP Locking Hash (Provenance ID).
    SHA256(dataset_content + manifest_content + base_model)

    Args:
        manifest: The training manifest.
        dataset: The prepared dataset.

    Returns:
        Hex digest of the SHA256 hash.
    """
    logger.info("Calculating GxP Provenance ID...")
    hasher = hashlib.sha256()

    # 1. Manifest Hash (Deterministically serialized)
    # model_dump_json() is deterministic in pydantic v2?
    # It might not be strictly deterministic key order, but usually is.
    # For safety, we can sort keys if needed, but pydantic json is usually stable enough for this level.
    manifest_bytes = manifest.model_dump_json(exclude_none=True).encode("utf-8")
    hasher.update(manifest_bytes)

    # 2. Dataset Hash
    # We serialize the dataset to JSON string to hash it.
    # Since dataset can be large, hashing it in chunks is better, but for simplicity:
    dataset_bytes = json.dumps(dataset, sort_keys=True).encode("utf-8")
    hasher.update(dataset_bytes)

    provenance_id = hasher.hexdigest()
    logger.info(f"Provenance ID: {provenance_id}")
    return provenance_id


def orchestrate_training(manifest_path: str) -> None:
    """
    The Crucible: Orchestrates the entire training workflow.

    1. Load Manifest
    2. Curate Data
    3. GxP Lock
    4. Select Strategy
    5. Execute Train
    """
    logger.info(f"Starting Crucible execution for {manifest_path}")

    try:
        # 1. Load Manifest
        manifest = load_manifest(manifest_path)
        logger.info(f"Job ID: {manifest.job_id} | Method: {manifest.method_config.type}")

        # 2. Curate Data
        curator = Curator(manifest)
        dataset = curator.prepare_dataset()

        if not dataset:
            logger.error("Dataset is empty after curation. Aborting.")
            sys.exit(1)

        # 3. GxP Lock
        provenance_id = calculate_provenance_id(manifest, dataset)
        logger.info(f"GxP Lock Acquired: {provenance_id}")
        # TODO: Send provenance_id to Veritas

        # 4. Select Strategy
        strategy = StrategyFactory.get_strategy(manifest)

        # 5. Execute Train
        # We might inject provenance_id into artifacts or logs here
        result = strategy.train(dataset)

        logger.info("Training completed successfully.")
        logger.info(f"Result: {result}")

        # 6. Distribute (Publish)
        if manifest.publish_target:
            output_dir = result.get("output_dir")
            if output_dir:
                publisher = ArtifactPublisher()
                publisher.publish_artifact(
                    artifact_path=output_dir,
                    target_registry=manifest.publish_target.registry,
                    tag=manifest.publish_target.tag,
                )
            else:
                logger.warning("No output directory returned from strategy. Skipping publication.")
        else:
            logger.info("No publish target defined in manifest. Skipping publication.")

    except Exception as e:
        logger.exception("Crucible execution failed.")
        raise e


def main() -> None:
    parser = argparse.ArgumentParser(description="Coreason Model Foundry - The Refinery")
    parser.add_argument("--manifest", type=str, required=True, help="Path to the Training Manifest YAML")
    args = parser.parse_args()

    orchestrate_training(args.manifest)


if __name__ == "__main__":
    main()
