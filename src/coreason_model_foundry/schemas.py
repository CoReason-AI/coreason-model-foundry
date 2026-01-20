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
    """Enumeration of supported training strategies."""

    QLORA = "qlora"
    DORA = "dora"
    ORPO = "orpo"


class MethodConfig(BaseModel):
    """Configuration for the training method."""

    type: MethodType
    rank: int = Field(..., gt=0, description="LoRA Rank")
    alpha: int = Field(..., gt=0, description="LoRA Alpha")
    target_modules: List[str] = Field(..., min_length=1, description="List of target modules for LoRA")
    strict_hardware_check: bool = Field(True, description="Fail on hardware mismatch if True")


class DatasetConfig(BaseModel):
    """Configuration for dataset selection and processing."""

    ref: str = Field(..., description="Reference URI for the dataset (e.g. synthesis://...)")
    sem_dedup: bool = Field(False, description="Enable Semantic Deduplication")
    dedup_threshold: Optional[float] = Field(
        0.95, ge=0.0, le=1.0, description="Cosine similarity threshold for deduplication"
    )


class ComputeConfig(BaseModel):
    """Configuration for computational resources and training hyperparameters."""

    batch_size: int = Field(..., gt=0, description="Training batch size per device")
    grad_accum: int = Field(..., gt=0, description="Gradient accumulation steps")
    context_window: int = Field(..., gt=0, description="Model context window size")
    quantization: Literal["4bit", "8bit", "none"] = Field("4bit", description="Quantization level")


class PublishConfig(BaseModel):
    """Configuration for artifact publishing."""

    registry: str = Field(..., description="Target registry URI")
    tag: str = Field(..., description="Version tag for the artifact")
    trigger_deployment: bool = Field(False, description="Whether to trigger immediate deployment")


class TrainingManifest(BaseModel):
    """Root configuration object for a training job."""

    job_id: str = Field(..., description="Unique identifier for the training job")
    base_model: str = Field(..., description="HuggingFace model ID or path")
    method_config: MethodConfig = Field(..., description="Training strategy configuration")
    dataset: DatasetConfig = Field(..., description="Dataset configuration")
    compute: ComputeConfig = Field(..., description="Compute configuration")
    publish_target: Optional[PublishConfig] = Field(None, description="Artifact publishing configuration")


class MergeMethod(str, Enum):
    """Enumeration of supported merge algorithms."""

    DARE_TIES = "dare_ties"


class ModelParameters(BaseModel):
    """Parameters for a single model in a merge recipe."""

    weight: float = Field(..., ge=0.0, description="Weight of the model in the merge")
    density: float = Field(..., ge=0.0, le=1.0, description="Density of the model parameters to keep")


class ModelEntry(BaseModel):
    """Definition of a model inclusion in a merge recipe."""

    model: str = Field(..., description="Model ID or path")
    parameters: ModelParameters = Field(..., description="Merge parameters for this model")


class MergeRecipe(BaseModel):
    """Configuration for a model merge job."""

    job_id: str = Field(..., description="Unique identifier for the merge job")
    merge_method: MergeMethod = Field(..., description="Algorithm to use for merging")
    base_model: str = Field(..., description="Base model to merge adapters into")
    models: List[ModelEntry] = Field(..., description="List of models/adapters to merge")
    dtype: Literal["float16", "bfloat16", "float32"] = Field(..., description="Data type for the merged model")
