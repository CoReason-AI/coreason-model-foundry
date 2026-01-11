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

from coreason_model_foundry.schemas import MethodType
from utils.logger import logger


class DataFormatter:
    """
    Formats data according to the training method requirements.
    """

    @staticmethod
    def format_and_validate(data: List[Dict[str, Any]], method: MethodType) -> List[Dict[str, Any]]:
        """
        Validates structure and standardizes format.

        Args:
            data: Raw data list.
            method: The training method (QLORA/DORA -> SFT format, ORPO -> Preference format).

        Returns:
            Formatted data list.

        Raises:
            ValueError: If required columns are missing.
        """
        logger.info(f"Formatting data for method: {method}")

        formatted_data = []

        if method in [MethodType.QLORA, MethodType.DORA]:
            # SFT Format: { instruction, input, output }
            # We expect source to have these or reasonable mappings.
            # Strict validation for now based on PRD: "Formats data as { instruction, input, output }"

            required_keys = {"instruction", "output"}  # input can be optional/empty string

            for idx, item in enumerate(data):
                if not required_keys.issubset(item.keys()):
                    raise ValueError(f"Row {idx} missing required keys for SFT: {required_keys - item.keys()}")

                formatted_data.append(
                    {"instruction": item["instruction"], "input": item.get("input", ""), "output": item["output"]}
                )

        elif method == MethodType.ORPO:
            # ORPO Format: { prompt, chosen, rejected }
            required_keys = {"prompt", "chosen", "rejected"}

            for idx, item in enumerate(data):
                if not required_keys.issubset(item.keys()):
                    raise ValueError(f"Row {idx} missing required keys for ORPO: {required_keys - item.keys()}")

                formatted_data.append(
                    {"prompt": item["prompt"], "chosen": item["chosen"], "rejected": item["rejected"]}
                )

        else:
            # Should be covered by schemas, but good safety
            raise ValueError(f"Unknown format requirement for method {method}")

        return formatted_data
