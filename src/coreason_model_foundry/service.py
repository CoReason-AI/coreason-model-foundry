# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

import contextvars
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
import anyio
import httpx
import yaml
from anyio import to_thread

from coreason_model_foundry.curator.main import Curator
from coreason_model_foundry.publisher import ArtifactPublisher
from coreason_model_foundry.schemas import TrainingManifest
from coreason_model_foundry.strategies.factory import StrategyFactory
from utils.logger import logger

# Replace threading.local with contextvars
request_id_ctx: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar("request_id", default=None)


class ModelFoundryServiceAsync:
    """Async Core Service for Model Foundry.

    Handles the lifecycle of resources and orchestrates the training workflow asynchronously.
    """

    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        """Initializes the service.

        Args:
            client: Optional external httpx.AsyncClient. If None, one will be created.
        """
        self._internal_client = client is None
        self._client = client or httpx.AsyncClient()
        self._publisher = ArtifactPublisher()

    async def __aenter__(self) -> "ModelFoundryServiceAsync":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._internal_client:
            await self._client.aclose()

    async def load_manifest(self, manifest_path: Path) -> TrainingManifest:
        """Loads and validates the Training Manifest asynchronously.

        Args:
            manifest_path: Path to the YAML file.

        Returns:
            TrainingManifest: The validated manifest.
        """
        if not await to_thread.run_sync(manifest_path.exists):
            raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

        logger.info(f"Loading manifest from {manifest_path}")
        async with aiofiles.open(manifest_path, "r", encoding="utf-8") as f:
            content = await f.read()
            # safe_load is CPU bound
            data = await to_thread.run_sync(yaml.safe_load, content)

        return TrainingManifest(**data)

    async def calculate_provenance_id(self, manifest: TrainingManifest, dataset: List[Dict[str, Any]]) -> str:
        """Calculates GxP Provenance ID asynchronously."""

        def _calc() -> str:
            logger.info("Calculating GxP Provenance ID...")
            hasher = hashlib.sha256()
            manifest_bytes = manifest.model_dump_json(exclude_none=True).encode("utf-8")
            hasher.update(manifest_bytes)
            dataset_bytes = json.dumps(dataset, sort_keys=True).encode("utf-8")
            hasher.update(dataset_bytes)
            provenance_id = hasher.hexdigest()
            logger.info(f"Provenance ID: {provenance_id}")
            return provenance_id

        return await to_thread.run_sync(_calc)

    async def orchestrate_training(self, manifest_path: Path) -> None:
        """Orchestrates the training workflow asynchronously."""
        logger.info(f"Starting Crucible execution for {manifest_path}")

        try:
            # 1. Load Manifest
            manifest = await self.load_manifest(manifest_path)
            logger.info(f"Job ID: {manifest.job_id} | Method: {manifest.method_config.type}")

            # 2. Curate Data
            # Curator operations might do I/O, but currently they are synchronous.
            # We wrap them in to_thread.run_sync for now if they are blocking.
            # However, looking at Curator, it uses DataResolver which reads files.
            # Ideally Curator should be refactored to be async, but per instructions,
            # we wrap CPU/Sync logic.
            curator = Curator(manifest)
            dataset = await to_thread.run_sync(curator.prepare_dataset)

            if not dataset:
                logger.error("Dataset is empty after curation. Aborting.")
                sys.exit(1)

            # 3. GxP Lock
            provenance_id = await self.calculate_provenance_id(manifest, dataset)
            logger.info(f"GxP Lock Acquired: {provenance_id}")

            # 4. Select Strategy
            strategy = StrategyFactory.get_strategy(manifest)

            # 5. Execute Train
            # Training is definitely heavy and blocking.
            result = await to_thread.run_sync(strategy.train, dataset)

            logger.info("Training completed successfully.")
            logger.info(f"Result: {result}")

            # 6. Distribute (Publish)
            if manifest.publish_target:
                output_dir = result.get("output_dir")
                if output_dir:
                    # Publisher might need network I/O.
                    # Ideally we'd use self._client if publisher supported async.
                    # Current publisher is sync.
                    # Note: to_thread.run_sync calls the function with *args.
                    # It seems `self._publisher.publish_artifact` is being called with positional args.
                    await to_thread.run_sync(
                        self._publisher.publish_artifact,
                        output_dir,
                        manifest.publish_target.registry,
                        manifest.publish_target.tag,
                    )
                else:
                    logger.warning("No output directory returned from strategy. Skipping publication.")
            else:
                logger.info("No publish target defined in manifest. Skipping publication.")

        except Exception as e:
            logger.exception("Crucible execution failed.")
            raise e


class ModelFoundryService:
    """Sync Facade for Model Foundry Service.

    Wraps the Async Core Service to provide synchronous access.
    """

    def __init__(self, client: Optional[httpx.AsyncClient] = None):
        self._async = ModelFoundryServiceAsync(client)

    def __enter__(self) -> "ModelFoundryService":
        # We start the async context in a way that we can clean up in __exit__
        # But wait, anyio.run usually blocks.
        # The prompt pattern says:
        # def __enter__(self): return self
        # def __exit__(self, *args): anyio.run(self._async.__aexit__, *args)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        anyio.run(self._async.__aexit__, exc_type, exc_val, exc_tb)

    def orchestrate_training(self, manifest_path: Path | str) -> None:
        """Orchestrates training synchronously."""
        if isinstance(manifest_path, str):
            manifest_path = Path(manifest_path)
        return anyio.run(self._async.orchestrate_training, manifest_path)
