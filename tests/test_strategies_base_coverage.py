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
from unittest.mock import MagicMock

from coreason_model_foundry.schemas import TrainingManifest
from coreason_model_foundry.strategies.base import TrainingStrategy


class ConcreteStrategy(TrainingStrategy):
    """Concrete implementation for testing base class."""

    def validate(self) -> None:
        self.validate_environment()

    def train(self, train_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        return {}


def test_base_validate_environment_coverage() -> None:
    """Test that base validate_environment can be called (coverage)."""
    manifest = MagicMock(spec=TrainingManifest)
    strategy = ConcreteStrategy(manifest)

    # Should not raise
    strategy.validate_environment()
