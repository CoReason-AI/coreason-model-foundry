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
from utils.logger import logger

try:
    import torch
except ImportError:
    torch = None

try:
    from unsloth import FastLanguageModel, PatchDPOTrainer
except ImportError:
    FastLanguageModel = None
    PatchDPOTrainer = None
    logger.warning("Unsloth not found. ORPO training will fail if executed on this environment.")


class ORPOStrategy(TrainingStrategy):
    """
    Implementation of ORPO (Odds Ratio Preference Optimization).
    Best for safety, alignment, and chat tasks.
    Requires Triplet Data.
    """

    MIN_VRAM_GB = 24
    BYTES_PER_GB = 1024**3

    def validate(self) -> None:
        """
        Validates hardware constraints for ORPO.
        Constraint: VRAM >= 24GB unless gradient checkpointing is strictly managed (but here we Fail Fast).
        """
        logger.info("Validating ORPO Strategy requirements.")

        if self.manifest.method_config.type != "orpo":
            # Should technically be checked by Factory, but good for safety
            raise ValueError("Invalid strategy type for ORPOStrategy")

        # Check Unsloth
        if FastLanguageModel is None:
            raise RuntimeError("Unsloth is required for ORPO strategy but is not installed.")

        # Check Hardware
        if torch and torch.cuda.is_available():
            gpu_idx = 0  # Default to 0 for now
            try:
                device_props = torch.cuda.get_device_properties(gpu_idx)
                total_memory = device_props.total_memory
                total_memory_gb = total_memory / self.BYTES_PER_GB

                logger.info(f"Detected VRAM: {total_memory_gb:.2f} GB")

                strict_check = getattr(self.manifest.method_config, "strict_hardware_check", True)

                # PRD: If VRAM < 24GB: Fail Fast
                # (unless allow_gradient_checkpointing is forced, which implies strict_check=False)
                if total_memory_gb < self.MIN_VRAM_GB:
                    if strict_check:
                        msg = (
                            f"Insufficient VRAM for ORPO. Detected {total_memory_gb:.2f}GB, required "
                            f"{self.MIN_VRAM_GB}GB. Upgrade node or disable strict hardware checks."
                        )
                        raise RuntimeError(msg)
                    else:
                        logger.warning(
                            f"VRAM {total_memory_gb:.2f}GB is below recommended {self.MIN_VRAM_GB}GB for ORPO. "
                            "Proceeding due to disabled strict check, but OOM is likely."
                        )
            except Exception as e:
                # If we can't query device (e.g. wrong index), we might warn or fail.
                # Assuming index 0 exists if is_available is true.
                if isinstance(e, RuntimeError) and "Insufficient VRAM" in str(e):
                    raise e
                logger.warning(f"Could not query GPU properties: {e}")
        else:
            logger.warning("No CUDA device detected. Validation skipped (assuming CPU/Mock environment).")

    def train(self, train_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Executes ORPO Training.
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
    try:
        import torch

        # Cast to bool explicitly to satisfy Mypy
        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False
