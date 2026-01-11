# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

from typing import Type

from coreason_model_foundry.schemas import MethodType, TrainingManifest
from coreason_model_foundry.strategies.base import TrainingStrategy
from coreason_model_foundry.strategies.dora import DoRAStrategy
from coreason_model_foundry.strategies.orpo import ORPOStrategy
from coreason_model_foundry.strategies.qlora import QLoRAStrategy


class StrategyFactory:
    """
    Factory to instantiate the correct TrainingStrategy based on the manifest.
    """

    _REGISTRY: dict[MethodType, Type[TrainingStrategy]] = {
        MethodType.QLORA: QLoRAStrategy,
        MethodType.DORA: DoRAStrategy,
        MethodType.ORPO: ORPOStrategy,
    }

    @classmethod
    def get_strategy(cls, manifest: TrainingManifest) -> TrainingStrategy:
        """
        Returns an instance of the appropriate strategy.

        Args:
            manifest: The TrainingManifest containing the method configuration.

        Returns:
            An instance of a subclass of TrainingStrategy.

        Raises:
            ValueError: If the method type is not supported.
        """
        method_type = manifest.method_config.type
        strategy_cls = cls._REGISTRY.get(method_type)

        if not strategy_cls:
            raise ValueError(f"Unsupported training method: {method_type}")

        strategy = strategy_cls(manifest)
        strategy.validate()
        return strategy
