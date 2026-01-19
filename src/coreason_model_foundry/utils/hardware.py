# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

from typing import Dict

from utils.logger import logger


class HardwareIncompatibleError(RuntimeError):
    """Raised when the hardware does not meet the requirements for the strategy."""

    pass


def get_gpu_memory_info() -> Dict[int, float]:
    """
    Returns a dictionary mapping GPU device IDs to their total memory in GB.
    Returns an empty dict if CUDA is not available.
    """
    try:
        import torch
    except ImportError:
        logger.warning("Torch not installed. Assuming no GPU.")
        return {}

    if not torch.cuda.is_available():
        logger.info("CUDA not available. No GPU memory info.")
        return {}

    info = {}
    device_count = torch.cuda.device_count()
    for i in range(device_count):
        try:
            # We use total_memory from get_device_properties
            props = torch.cuda.get_device_properties(i)
            total_mem_gb = props.total_memory / (1024**3)
            info[i] = total_mem_gb
        except Exception as e:
            logger.warning(f"Failed to get properties for device {i}: {e}")

    return info


def check_vram_compatibility(required_gb: float, device_id: int = 0) -> None:
    """
    Checks if the specified GPU has enough VRAM.

    Args:
        required_gb: The minimum required VRAM in GB.
        device_id: The GPU device ID to check (default 0).

    Raises:
        HardwareIncompatibleError: If VRAM is insufficient.
    """
    try:
        import torch
    except ImportError:
        logger.warning("Torch not installed. Skipping VRAM check.")
        return

    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Skipping VRAM check (running on CPU?).")
        # If strict checking is required for CPU, caller should handle "no gpu" before calling this
        # or we could raise error if GPU is strictly required.
        # But for compatibility, we just log warning here as "check_vram" implies checking GPU VRAM.
        return

    mem_info = get_gpu_memory_info()
    if device_id not in mem_info:
        # If device_id is not found but CUDA is available, that's an issue (e.g. wrong index)
        # But if mem_info is empty, we already returned.
        logger.warning(f"Device {device_id} not found in memory info. Available: {list(mem_info.keys())}")
        return

    available_gb = mem_info[device_id]

    # We compare with a small buffer? The prompt says "if VRAM < 24GB".
    # Usually 24GB GPU has slightly less usuable, but total_memory reports physical.
    # We'll stick to strict comparison against total physical memory.

    logger.info(f"Checking VRAM for device {device_id}: Available={available_gb:.2f}GB, Required={required_gb}GB")

    if available_gb < required_gb:
        msg = (
            f"Insufficient VRAM on device {device_id}. "
            f"Required: {required_gb}GB, Available: {available_gb:.2f}GB. "
            "Try using quantization, gradient checkpointing, or a larger node."
        )
        logger.error(msg)
        raise HardwareIncompatibleError(msg)

    logger.info("VRAM check passed.")
