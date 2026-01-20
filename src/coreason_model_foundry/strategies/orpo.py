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

from datasets import Dataset
from transformers import TrainingArguments

# We use trl's ORPOTrainer. Unsloth patches it or provides optimized kernels.
# The PRD mentions UnslothORPOTrainer, but in practice one often uses trl.ORPOTrainer with Unsloth models.
# We will use ORPOTrainer from trl for the interface.
from trl import ORPOTrainer

from coreason_model_foundry.strategies.base import TrainingStrategy
from coreason_model_foundry.utils.hardware import check_vram_compatibility
from utils.logger import logger

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from unsloth import FastLanguageModel, PatchDPOTrainer
except ImportError:
    FastLanguageModel = None
    PatchDPOTrainer = None
    logger.warning("Unsloth not found. ORPO training will fail if executed on this environment.")


class ORPOStrategy(TrainingStrategy):
    """Implementation of ORPO (Odds Ratio Preference Optimization).

    Best for safety, alignment, and chat tasks. Requires Triplet Data (prompt, chosen, rejected).
    Implements strict hardware validation for high-memory configurations.
    """

    MIN_VRAM_GB = 24

    def validate_environment(self) -> None:
        """Validates hardware constraints for ORPO.

        Rule: If quantization == 'none' AND vram < 24GB -> Raise HardwareIncompatibleError.

        Raises:
            HardwareIncompatibleError: If VRAM is insufficient.
        """
        logger.info("Validating ORPO Environment...")

        quantization = self.manifest.compute.quantization
        logger.info(f"Quantization mode: {quantization}")

        # Check if we need to enforce the 24GB limit
        # "If quantization == 'none' AND vram < 24GB"
        if quantization == "none":
            # We assume 'none' means full precision or BF16, which is heavy.
            # We enforce 24GB VRAM.
            logger.info(f"Full precision (quantization='none') detected. Enforcing {self.MIN_VRAM_GB}GB VRAM check.")
            check_vram_compatibility(required_gb=self.MIN_VRAM_GB)
        else:
            # For 4bit/8bit, we might be more lenient, or strict check applies differently.
            # The user instruction was specific to quantization == 'none'.
            # However, we can still run a check if strict hardware check is enabled?
            # For now, we only enforce what was requested to avoid regression on other configs.
            pass

    def validate(self) -> None:
        """Validates if the current environment and manifest are suitable for this strategy.

        Raises:
            ValueError: If strategy type is incorrect.
            RuntimeError: If Unsloth is missing.
            HardwareIncompatibleError: If hardware checks fail.
        """
        logger.info("Validating ORPO Strategy requirements.")

        if self.manifest.method_config.type != "orpo":
            raise ValueError("Invalid strategy type for ORPOStrategy")

        # Check Unsloth
        if FastLanguageModel is None:
            raise RuntimeError("Unsloth is required for ORPO strategy but is not installed.")

        # Check Hardware Environment
        self.validate_environment()

    def train(self, train_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Executes ORPO Training.

        Args:
            train_dataset: The processed dataset ready for training.

        Returns:
            Dict[str, Any]: Dictionary containing artifacts paths or execution status.

        Raises:
            ValueError: If dataset is empty.
            RuntimeError: If Unsloth is missing.
        """
        logger.info(f"Initializing ORPO training for job {self.manifest.job_id}")

        if not train_dataset:
            raise ValueError("Training dataset is empty.")

        # 0. Patching (Specific to Unsloth DPO/ORPO)
        if PatchDPOTrainer is not None:
            PatchDPOTrainer()

        # 1. Load Model & Tokenizer
        model_name = self.manifest.base_model
        max_seq_length = self.manifest.compute.context_window
        load_in_4bit = self.manifest.compute.quantization == "4bit"

        logger.info(f"Loading Base Model: {model_name} (4bit={load_in_4bit})")

        # Typing safety check (validate() should have caught this, but for static analysis)
        if FastLanguageModel is None:
            raise RuntimeError("Unsloth is required.")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )

        # 2. Add LoRA Adapters
        # ORPO is a finetuning method that often works on top of LoRA
        logger.info("Applying LoRA Adapters for ORPO...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=self.manifest.method_config.rank,
            target_modules=self.manifest.method_config.target_modules,
            lora_alpha=self.manifest.method_config.alpha,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=3407,
            use_rslora=False,
            loftq_config=None,
        )

        # 3. Prepare Dataset
        # ORPO expects: prompt, chosen, rejected
        # train_dataset is List[Dict]. We assume it is already formatted by Curator.
        # We need to verify formatting just in case or trust Curator.
        # Curator DataFormatter enforces {prompt, chosen, rejected} for ORPO.

        logger.info(f"Converting {len(train_dataset)} examples to HF Dataset.")
        hf_dataset = Dataset.from_list(train_dataset)

        # 4. Setup Trainer
        output_dir = f"artifacts/{self.manifest.job_id}"
        training_args = TrainingArguments(
            per_device_train_batch_size=self.manifest.compute.batch_size,
            gradient_accumulation_steps=self.manifest.compute.grad_accum,
            warmup_steps=5,
            max_steps=0,
            num_train_epochs=1,
            learning_rate=5e-6,  # Lower LR for ORPO usually
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
            remove_unused_columns=False,  # Important for ORPO datasets sometimes
        )

        trainer = ORPOTrainer(
            model=model,
            train_dataset=hf_dataset,
            tokenizer=tokenizer,
            args=training_args,
            max_length=max_seq_length,
            max_prompt_length=max_seq_length // 2,  # Heuristic
        )

        # 5. Train
        logger.info("Starting Training Loop...")
        trainer.train()

        # 6. Save Artifacts
        logger.info(f"Saving artifacts to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        return {
            "status": "success",
            "output_dir": output_dir,
            "job_id": self.manifest.job_id,
            "strategy": "orpo",
        }


def is_bfloat16_supported() -> bool:
    """Checks if bfloat16 is supported on the current device."""
    try:
        import torch

        # Cast to bool explicitly to satisfy Mypy
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False
