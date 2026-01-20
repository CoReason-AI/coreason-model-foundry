# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

from typing import Any, Dict, List

from coreason_model_foundry.curator.deduplicator import SemDeDup
from coreason_model_foundry.curator.formatter import DataFormatter
from coreason_model_foundry.curator.resolver import DataResolver
from coreason_model_foundry.schemas import MethodType, TrainingManifest
from utils.logger import logger


class Curator:
    """The Data Curator.

    Orchestrates the data pipeline, including resolution, validation, formatting,
    and semantic deduplication.
    """

    def __init__(self, manifest: TrainingManifest):
        """Initializes the Curator with the given training manifest.

        Args:
            manifest: The training manifest configuration.
        """
        self.manifest = manifest
        self.resolver = DataResolver()
        self.deduplicator = SemDeDup(threshold=self.manifest.dataset.dedup_threshold or 0.95)

    def prepare_dataset(self) -> List[Dict[str, Any]]:
        """Runs the full curation pipeline.

        1. Resolves data from the URI (DataResolver).
        2. Formats and validates data (DataFormatter).
        3. Applies Semantic Deduplication if enabled (SemDeDup).

        Returns:
            List[Dict[str, Any]]: A list of formatted, deduped data dictionaries ready for training.
        """
        logger.info(f"Starting curation for job {self.manifest.job_id}")

        # 1. Resolve Data
        raw_data = self.resolver.resolve(self.manifest.dataset.ref)
        logger.info(f"Resolved {len(raw_data)} raw examples.")

        # 2. Format & Validate
        formatted_data = DataFormatter.format_and_validate(raw_data, self.manifest.method_config.type)

        # 3. Deduplicate (if enabled)
        if self.manifest.dataset.sem_dedup:
            # Determine key fields for embedding based on method
            if self.manifest.method_config.type in [MethodType.QLORA, MethodType.DORA]:
                key_fields = ["instruction", "input"]
            else:  # ORPO
                key_fields = ["prompt"]

            final_data = self.deduplicator.prune(formatted_data, key_fields=key_fields)
        else:
            final_data = formatted_data
            logger.info("Semantic deduplication skipped (disabled in manifest).")

        return final_data
