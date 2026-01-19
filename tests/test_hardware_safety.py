# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

import sys
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

from coreason_model_foundry.utils.hardware import (
    HardwareIncompatibleError,
    check_vram_compatibility,
    get_gpu_memory_info,
)


@pytest.fixture
def mock_torch() -> Generator[MagicMock, None, None]:
    with patch.dict(sys.modules, {"torch": MagicMock()}):
        import torch

        yield torch


def test_get_gpu_memory_info_no_cuda(mock_torch: MagicMock) -> None:
    """Test behavior when CUDA is not available."""
    mock_torch.cuda.is_available.return_value = False

    info = get_gpu_memory_info()
    assert info == {}
    mock_torch.cuda.device_count.assert_not_called()


def test_get_gpu_memory_info_single_gpu(mock_torch: MagicMock) -> None:
    """Test getting memory info for a single GPU."""
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1

    # Mock properties
    mock_props = MagicMock()
    # 24 GB in bytes
    mock_props.total_memory = 24 * (1024**3)
    mock_torch.cuda.get_device_properties.return_value = mock_props

    info = get_gpu_memory_info()
    assert len(info) == 1
    assert info[0] == 24.0
    mock_torch.cuda.get_device_properties.assert_called_with(0)


def test_get_gpu_memory_info_multi_gpu(mock_torch: MagicMock) -> None:
    """Test getting memory info for multiple GPUs."""
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 2

    # Mock properties side_effect
    prop1 = MagicMock()
    prop1.total_memory = 24 * (1024**3)
    prop2 = MagicMock()
    prop2.total_memory = 80 * (1024**3)

    mock_torch.cuda.get_device_properties.side_effect = [prop1, prop2]

    info = get_gpu_memory_info()
    assert len(info) == 2
    assert info[0] == 24.0
    assert info[1] == 80.0


def test_check_vram_compatibility_success(mock_torch: MagicMock) -> None:
    """Test successful VRAM check."""
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_props = MagicMock()
    mock_props.total_memory = 32 * (1024**3)  # 32 GB
    mock_torch.cuda.get_device_properties.return_value = mock_props

    # Should not raise
    check_vram_compatibility(24.0)


def test_check_vram_compatibility_failure(mock_torch: MagicMock) -> None:
    """Test failure when VRAM is insufficient."""
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_props = MagicMock()
    mock_props.total_memory = 16 * (1024**3)  # 16 GB
    mock_torch.cuda.get_device_properties.return_value = mock_props

    with pytest.raises(HardwareIncompatibleError) as exc:
        check_vram_compatibility(24.0)

    assert "Insufficient VRAM" in str(exc.value)
    assert "Required: 24.0GB" in str(exc.value)
    assert "Available: 16.00GB" in str(exc.value)


def test_check_vram_compatibility_cpu_only(mock_torch: MagicMock) -> None:
    """Test VRAM check on CPU only (should warn but not crash/raise for now)."""
    mock_torch.cuda.is_available.return_value = False

    # Should return None/log warning
    check_vram_compatibility(24.0)


def test_check_vram_compatibility_device_not_found(mock_torch: MagicMock) -> None:
    """Test checking a device ID that doesn't exist."""
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_props = MagicMock()
    mock_props.total_memory = 24 * (1024**3)
    mock_torch.cuda.get_device_properties.return_value = mock_props

    # get_gpu_memory_info iterates device_count=1 (index 0)
    # So index 1 won't be in the dict

    check_vram_compatibility(24.0, device_id=1)
    # Should log warning and return (not raise) based on current impl
