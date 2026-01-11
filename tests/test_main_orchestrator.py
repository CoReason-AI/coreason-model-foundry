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
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from coreason_model_foundry.main import (
    calculate_provenance_id,
    load_manifest,
    main,
    orchestrate_training,
)
from coreason_model_foundry.schemas import TrainingManifest

# Sample Manifest Data
MANIFEST_YAML = """
job_id: "test-job-001"
base_model: "meta-llama/Meta-Llama-3-8B"
method_config:
  type: "dora"
  rank: 64
  alpha: 16
  target_modules: ["q_proj", "v_proj"]
dataset:
  ref: "synthesis://test_dataset"
  sem_dedup: true
  dedup_threshold: 0.95
compute:
  batch_size: 4
  grad_accum: 4
  context_window: 8192
  quantization: "4bit"
"""


@pytest.fixture
def mock_manifest_file(tmp_path: Path) -> str:
    p = tmp_path / "manifest.yaml"
    p.write_text(MANIFEST_YAML)
    return str(p)


def test_load_manifest_valid(mock_manifest_file: str) -> None:
    manifest = load_manifest(mock_manifest_file)
    assert isinstance(manifest, TrainingManifest)
    assert manifest.job_id == "test-job-001"
    assert manifest.method_config.type == "dora"


def test_load_manifest_not_found() -> None:
    with pytest.raises(FileNotFoundError):
        load_manifest("non_existent.yaml")


def test_load_manifest_invalid_yaml(tmp_path: Path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("job_id: [broken yaml")
    with pytest.raises(yaml.YAMLError):
        load_manifest(str(p))


def test_calculate_provenance_id() -> None:
    manifest = MagicMock()
    manifest.model_dump_json.return_value = '{"job": "1"}'
    dataset = [{"data": "1"}]

    # Check consistency
    pid1 = calculate_provenance_id(manifest, dataset)
    pid2 = calculate_provenance_id(manifest, dataset)
    assert pid1 == pid2
    assert len(pid1) == 64  # SHA256 hex digest length


def test_calculate_provenance_id_sensitivity() -> None:
    """Complex Scenario: Verify hash sensitivity to changes."""
    manifest_a = MagicMock()
    manifest_a.model_dump_json.return_value = '{"job": "1"}'

    manifest_b = MagicMock()
    manifest_b.model_dump_json.return_value = '{"job": "2"}'

    dataset_a = [{"data": "1"}]
    dataset_b = [{"data": "2"}]

    pid_aa = calculate_provenance_id(manifest_a, dataset_a)
    pid_ab = calculate_provenance_id(manifest_a, dataset_b)
    pid_ba = calculate_provenance_id(manifest_b, dataset_a)

    assert pid_aa != pid_ab, "Hash should change if dataset changes"
    assert pid_aa != pid_ba, "Hash should change if manifest changes"
    assert pid_ab != pid_ba


@patch("coreason_model_foundry.main.Curator")
@patch("coreason_model_foundry.main.StrategyFactory")
@patch("coreason_model_foundry.main.load_manifest")
def test_orchestrate_training_flow(mock_load: MagicMock, mock_factory: MagicMock, mock_curator_cls: MagicMock) -> None:
    # Setup Mocks
    mock_manifest = MagicMock()
    mock_manifest.job_id = "test-job"
    # Ensure nested attributes are mocked
    mock_manifest.method_config.type = "dora"
    # Ensure model_dump_json returns a string so encode works
    mock_manifest.model_dump_json.return_value = '{"job_id": "test-job"}'
    mock_load.return_value = mock_manifest

    mock_curator_instance = mock_curator_cls.return_value
    mock_curator_instance.prepare_dataset.return_value = [{"instruction": "foo", "output": "bar"}]

    mock_strategy = MagicMock()
    mock_strategy.train.return_value = {"status": "success", "output_dir": "artifacts/test"}
    mock_factory.get_strategy.return_value = mock_strategy

    # Execute
    orchestrate_training("dummy_path.yaml")

    # Verify Interactions
    mock_load.assert_called_once_with("dummy_path.yaml")

    # Curator
    mock_curator_cls.assert_called_once_with(mock_manifest)
    mock_curator_instance.prepare_dataset.assert_called_once()

    # Strategy
    mock_factory.get_strategy.assert_called_once_with(mock_manifest)
    mock_strategy.train.assert_called_once()


@patch("coreason_model_foundry.main.Curator")
@patch("coreason_model_foundry.main.load_manifest")
def test_orchestrate_training_empty_dataset(mock_load: MagicMock, mock_curator_cls: MagicMock) -> None:
    # Setup
    mock_load.return_value = MagicMock()
    mock_curator_instance = mock_curator_cls.return_value
    mock_curator_instance.prepare_dataset.return_value = []  # Empty dataset

    # Expect SystemExit(1)
    with pytest.raises(SystemExit) as exc:
        orchestrate_training("dummy.yaml")

    assert exc.value.code == 1


@patch("coreason_model_foundry.main.load_manifest")
def test_orchestrate_training_exception(mock_load: MagicMock) -> None:
    # Simulate an error during manifest loading
    mock_load.side_effect = ValueError("Critical Failure")

    with pytest.raises(ValueError, match="Critical Failure"):
        orchestrate_training("dummy.yaml")


@patch("coreason_model_foundry.main.orchestrate_training")
def test_main_cli(mock_orchestrate: MagicMock) -> None:
    test_args = ["main.py", "--manifest", "config.yaml"]
    with patch.object(sys, "argv", test_args):
        main()

    mock_orchestrate.assert_called_once_with("config.yaml")
