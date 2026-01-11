# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from coreason_model_foundry.schemas import TrainingManifest


class TrainingStrategy(ABC):
    """
    Abstract Base Class for training strategies.

    This enforces the interface for different training kernels (QLoRA, DoRA, ORPO).
    """

    def __init__(self, manifest: TrainingManifest):
        self.manifest = manifest

    @abstractmethod
    def validate(self) -> None:
        """
        Validates if the current environment and manifest are suitable for this strategy.
        Raises exceptions if constraints are not met.
        """
        pass  # pragma: no cover

    @abstractmethod
    def train(self, train_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes the training process.

        Args:
            train_dataset: The processed dataset ready for training.

        Returns:
            Dict containing artifacts paths or execution status.
        """
        pass  # pragma: no cover
