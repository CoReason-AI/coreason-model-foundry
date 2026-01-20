# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

import json
from pathlib import Path
from typing import Any, Dict, List

from utils.logger import logger


class DataResolver:
    """Resolves data references (URIs) to actual data.

    Supports local files for `synthesis://` scheme in testing/dev environments.
    """

    @staticmethod
    def resolve(uri: str) -> List[Dict[str, Any]]:
        """Resolves the given URI to a list of data dictionaries.

        Currently supports the `synthesis://` scheme by mapping it to a local
        file path (e.g., `synthesis://path/to/data` -> `path/to/data.json`).
        It attempts to append `.json` if missing and checks the current working directory.

        Args:
            uri: The URI to resolve (e.g., "synthesis://batch_clinical_reasoning").

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the data.

        Raises:
            FileNotFoundError: If the file referenced by the URI does not exist.
            ValueError: If the URI scheme is unsupported or the data format is invalid.
        """
        logger.info(f"Resolving data from URI: {uri}")

        if uri.startswith("synthesis://"):
            # Mock implementation: Map synthesis:// to a local file
            # In a real system, this would call an API or DB.
            # For this 'foundry', we treat it as looking up a file in a known directory
            # or simply stripping the scheme and expecting a path for testing.

            # For simplicity in this atomic unit, we'll implement a simple mapping
            # or treat the rest of the URI as a filename in a 'data' directory if it exists,
            # OR just assume the user maps it via a config (not present yet).

            # Let's treat "synthesis://path/to/data" as "tests/data/path/to/data.json" for now
            # or just generic file loading if it looks like a path.

            resource_name = uri.replace("synthesis://", "")

            # Try finding it in local path (useful for tests)
            # We'll assume the uri might point to a local json file for the mock
            # If the user provides an absolute path in the test, we use it.
            # But usually synthesis:// implies a logical name.

            # We will support a simple registry or environment variable based mapping later.
            # For now, let's assume it maps to a local file path provided in tests
            # OR we check a default location.

            # NOTE: For the purpose of this Atomic Unit and TDD, we will allow
            # the test to set up the file and pass a URI that helps us find it.
            # e.g. synthesis://tests/data/dummy.json

            file_path = Path(resource_name)
            if not file_path.suffix:
                file_path = file_path.with_suffix(".json")

            if not file_path.exists():
                # Fallback: check if it's relative to repo root
                file_path = Path.cwd() / resource_name
                if not file_path.suffix:
                    file_path = file_path.with_suffix(".json")

            if not file_path.exists():
                raise FileNotFoundError(f"Could not resolve synthesis resource: {resource_name}")

            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError(f"Data at {uri} must be a list of records.")

            logger.info(f"Loaded {len(data)} records from {file_path}")
            return data

        raise ValueError(f"Unsupported URI scheme in: {uri}")
