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

from coreason_model_foundry.strategies.base import TrainingStrategy
from utils.logger import logger


class QLoRAStrategy(TrainingStrategy):
    """
    Implementation of QLoRA.
    """

    def validate(self) -> None:
        if self.manifest.compute.quantization != "4bit":
            logger.warning("QLoRA usually requires 4bit quantization.")

    def train(self, train_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {"status": "mock_success", "strategy": "qlora"}
