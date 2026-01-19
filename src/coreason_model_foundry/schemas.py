# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

from enum import Enum
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class MethodType(str, Enum):
    QLORA = "qlora"
    DORA = "dora"
    ORPO = "orpo"


class MethodConfig(BaseModel):
    type: MethodType
    rank: int = Field(..., gt=0, description="LoRA Rank")
    alpha: int = Field(..., gt=0, description="LoRA Alpha")
    target_modules: List[str] = Field(..., min_length=1, description="List of target modules for LoRA")
    strict_hardware_check: bool = Field(True, description="Fail on hardware mismatch if True")


class DatasetConfig(BaseModel):
    ref: str
    sem_dedup: bool = False
    dedup_threshold: Optional[float] = Field(0.95, ge=0.0, le=1.0)


class ComputeConfig(BaseModel):
    batch_size: int = Field(..., gt=0)
    grad_accum: int = Field(..., gt=0)
    context_window: int = Field(..., gt=0)
    quantization: Literal["4bit", "8bit", "none"] = "4bit"


class TrainingManifest(BaseModel):
    job_id: str
    base_model: str
    method_config: MethodConfig
    dataset: DatasetConfig
    compute: ComputeConfig


class MergeMethod(str, Enum):
    DARE_TIES = "dare_ties"
    # Add other methods if needed, PRD only specifies dare_ties currently but implies others might exist in mergekit


class ModelParameters(BaseModel):
    weight: float = Field(..., ge=0.0)
    density: float = Field(..., ge=0.0, le=1.0)


class ModelEntry(BaseModel):
    model: str
    parameters: ModelParameters


class MergeRecipe(BaseModel):
    job_id: str
    merge_method: MergeMethod
    base_model: str
    models: List[ModelEntry]
    dtype: Literal["float16", "bfloat16", "float32"]
