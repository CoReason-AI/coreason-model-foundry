# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

"""coreason-model-foundry: The "Refinery" for post-training optimization.

This package serves as the industrial automation engine for training specialized
"Student Models". It orchestrates strategy selection (DoRA, ORPO, QLoRA), data pruning,
training via `unsloth`, and artifact distribution.
"""

__version__ = "0.1.0"
__author__ = "Gowtham A Rao"
__email__ = "gowtham.rao@coreason.ai"

from .service import ModelFoundryService, ModelFoundryServiceAsync

__all__ = ["ModelFoundryServiceAsync", "ModelFoundryService"]
