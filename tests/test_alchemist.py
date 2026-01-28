# Copyright (c) 2025 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.
#
# Source Code: https://github.com/CoReason-AI/coreason_model_foundry

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from coreason_identity.models import UserContext
from coreason_identity.types import SecretStr
from coreason_model_foundry.alchemist import Alchemist
from coreason_model_foundry.schemas import MergeMethod, MergeRecipe, ModelEntry, ModelParameters


@pytest.fixture
def mock_context() -> UserContext:
    return UserContext(user_id=SecretStr("test-user"), roles=["tester"])


@pytest.fixture
def sample_recipe() -> MergeRecipe:
    return MergeRecipe(
        job_id="test-merge-job",
        merge_method=MergeMethod.DARE_TIES,
        base_model="base-model-v1",
        models=[
            ModelEntry(model="adapter-logic", parameters=ModelParameters(weight=1.0, density=0.5)),
            ModelEntry(model="adapter-safety", parameters=ModelParameters(weight=0.5, density=0.7)),
        ],
        dtype="float16",
    )


class TestAlchemist:
    def test_build_dare_ties_config(self, sample_recipe: MergeRecipe) -> None:
        """Test that the DARE-TIES config dict is generated correctly."""
        alchemist = Alchemist()
        config = alchemist._build_dare_ties_config(sample_recipe)

        assert config["merge_method"] == "dare_ties"
        assert config["base_model"] == "base-model-v1"
        assert config["dtype"] == "float16"
        assert len(config["models"]) == 2

        # Check first model
        assert config["models"][0]["model"] == "adapter-logic"
        assert config["models"][0]["parameters"]["weight"] == 1.0
        assert config["models"][0]["parameters"]["density"] == 0.5

    @patch("subprocess.run")
    @patch("torch.cuda.is_available", return_value=False)
    def test_merge_execution_success(
        self,
        mock_cuda: MagicMock,
        mock_run: MagicMock,
        sample_recipe: MergeRecipe,
        tmp_path: Path,
        mock_context: UserContext,
    ) -> None:
        """Test successful execution of the merge process."""
        alchemist = Alchemist()
        output_dir = tmp_path / "output"

        # Execute
        result = alchemist.merge(sample_recipe, output_dir, context=mock_context)

        assert result == output_dir

        # Verify subprocess call
        assert mock_run.call_count == 1
        args, _ = mock_run.call_args
        cmd = args[0]

        assert cmd[0] == "mergekit-yaml"
        assert cmd[2] == str(output_dir)
        assert "--copy-tokenizer" in cmd
        assert "--cuda" not in cmd  # mocking no cuda

    @patch("subprocess.run")
    @patch("torch.cuda.is_available", return_value=True)
    def test_merge_execution_with_cuda(
        self,
        mock_cuda: MagicMock,
        mock_run: MagicMock,
        sample_recipe: MergeRecipe,
        tmp_path: Path,
        mock_context: UserContext,
    ) -> None:
        """Test execution with CUDA flag enabled."""
        alchemist = Alchemist()
        output_dir = tmp_path / "output"

        alchemist.merge(sample_recipe, output_dir, context=mock_context)

        args, _ = mock_run.call_args
        cmd = args[0]
        assert "--cuda" in cmd

    @patch("subprocess.run")
    def test_merge_execution_failure(
        self, mock_run: MagicMock, sample_recipe: MergeRecipe, tmp_path: Path, mock_context: UserContext
    ) -> None:
        """Test that subprocess failure raises RuntimeError."""
        # Simulate failure
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="mergekit-yaml", stderr="Some error occurred"
        )

        alchemist = Alchemist()
        output_dir = tmp_path / "output"

        with pytest.raises(RuntimeError) as exc:
            alchemist.merge(sample_recipe, output_dir, context=mock_context)

        assert "Merge failed" in str(exc.value)
        assert "Some error occurred" in str(exc.value)

    def test_unimplemented_method(self, sample_recipe: MergeRecipe) -> None:
        """Test that unknown methods raise NotImplementedError."""
        alchemist = Alchemist()
        # Mocking the recipe's merge_method attribute to bypass Pydantic enum validation
        # and test the NotImplementedError branch in Alchemist._build_config
        with patch.object(sample_recipe, "merge_method", "INVALID_METHOD"):
            with pytest.raises(NotImplementedError) as exc:
                alchemist._build_config(sample_recipe)

        assert "is not implemented" in str(exc.value)

    def test_temp_file_cleanup(self, sample_recipe: MergeRecipe, tmp_path: Path, mock_context: UserContext) -> None:
        """Verify that the temporary config file is cleaned up."""
        alchemist = Alchemist()
        output_dir = tmp_path / "output"

        with patch("pathlib.Path.unlink") as mock_unlink:
            # We also need to mock subprocess so it doesn't actually run/fail
            with patch("subprocess.run"):
                alchemist.merge(sample_recipe, output_dir, context=mock_context)

            mock_unlink.assert_called_once()

    @patch("subprocess.run")
    def test_merge_execution_missing_torch(
        self, mock_run: MagicMock, sample_recipe: MergeRecipe, tmp_path: Path, mock_context: UserContext
    ) -> None:
        """Test execution when torch is not installed (ImportError coverage)."""
        alchemist = Alchemist()
        output_dir = tmp_path / "output"

        # Simulate missing torch
        with patch.dict(sys.modules, {"torch": None}):
            alchemist.merge(sample_recipe, output_dir, context=mock_context)

        # Verify command does NOT have --cuda, and no exception was raised during check
        args, _ = mock_run.call_args
        cmd = args[0]
        assert "--cuda" not in cmd
