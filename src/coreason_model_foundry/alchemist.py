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
import tempfile
from pathlib import Path
from typing import Any, Dict

import yaml

from coreason_model_foundry.schemas import MergeMethod, MergeRecipe
from utils.logger import logger


class Alchemist:
    """
    The Alchemist: Orchestrates model merging using mergekit.
    """

    def merge(self, recipe: MergeRecipe, output_dir: Path) -> Path:
        """
        Executes the merge process based on the provided recipe.

        Args:
            recipe: The MergeRecipe configuration.
            output_dir: The directory to save the merged model.

        Returns:
            The path to the output directory.
        """
        logger.info(f"Initiating merge job {recipe.job_id} using {recipe.merge_method}")

        # 1. Build Config
        config_data = self._build_config(recipe)

        # 2. Write Config to Temp File
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
            yaml.dump(config_data, tmp)
            config_path = Path(tmp.name)

        try:
            # 3. Execute mergekit
            self._execute_mergekit(config_path, output_dir)
        finally:
            # 4. Cleanup
            if config_path.exists():
                config_path.unlink()

        return output_dir

    def _build_config(self, recipe: MergeRecipe) -> Dict[str, Any]:
        """Dispatches to the correct config builder."""
        if recipe.merge_method == MergeMethod.DARE_TIES:
            return self._build_dare_ties_config(recipe)
        else:
            raise NotImplementedError(f"Merge method {recipe.merge_method} is not implemented.")

    def _build_dare_ties_config(self, recipe: MergeRecipe) -> Dict[str, Any]:
        """
        Constructs the mergekit YAML structure for DARE-TIES.
        """
        models_config = []

        # Add models from recipe
        for model_entry in recipe.models:
            models_config.append(
                {
                    "model": model_entry.model,
                    "parameters": {
                        "weight": model_entry.parameters.weight,
                        "density": model_entry.parameters.density,
                    },
                }
            )

        return {
            "merge_method": "dare_ties",
            "base_model": recipe.base_model,
            "models": models_config,
            "dtype": recipe.dtype,
        }

    def _execute_mergekit(self, config_path: Path, output_dir: Path) -> None:
        """Runs the mergekit-yaml command."""
        cmd = [
            "mergekit-yaml",
            str(config_path),
            str(output_dir),
            "--copy-tokenizer",
        ]

        # Simple check for CUDA (mock-safe)
        try:
            import torch

            if torch.cuda.is_available():
                cmd.append("--cuda")
        except ImportError:
            pass

        logger.info(f"Executing: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Mergekit failed: {e.stderr}")
            raise RuntimeError(f"Merge failed: {e.stderr}") from e
