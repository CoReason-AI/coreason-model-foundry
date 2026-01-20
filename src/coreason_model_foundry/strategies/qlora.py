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
from trl import SFTTrainer

from coreason_model_foundry.strategies.base import TrainingStrategy
from utils.logger import logger

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from unsloth import FastLanguageModel
except ImportError:
    FastLanguageModel = None
    logger.warning("Unsloth not found. QLoRA training will fail if executed on this environment.")


class QLoRAStrategy(TrainingStrategy):
    """Implementation of QLoRA (Quantized Low-Rank Adaptation).

    Best for memory efficiency. Enforces 4-bit quantization and CUDA availability.
    """

    def validate_environment(self) -> None:
        """Validates hardware constraints for QLoRA.

        Rule: Must run on GPU (CUDA available).

        Raises:
            EnvironmentError: If CUDA is not available.
        """
        logger.info("Validating QLoRA Environment...")

        if torch is None or not torch.cuda.is_available():
            msg = "QLoRA requires a GPU environment (CUDA). CPU execution is not supported."
            logger.error(msg)
            raise EnvironmentError(msg)

    def validate(self) -> None:
        """Validates if the current environment and manifest are suitable for this strategy.

        Raises:
            RuntimeError: If Unsloth is missing.
            EnvironmentError: If environment checks fail.
        """
        logger.info("Validating QLoRA Strategy requirements.")

        if self.manifest.compute.quantization != "4bit":
            logger.warning("QLoRA usually requires 4bit quantization.")

        if FastLanguageModel is None:
            raise RuntimeError("Unsloth is required for QLoRA strategy but is not installed.")

        self.validate_environment()

    def train(self, train_dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Executes QLoRA Training.

        Args:
            train_dataset: The processed dataset ready for training.

        Returns:
            Dict[str, Any]: Dictionary containing artifacts paths or execution status.

        Raises:
            ValueError: If dataset is empty.
            RuntimeError: If Unsloth is missing.
        """
        logger.info(f"Initializing QLoRA training for job {self.manifest.job_id}")

        if not train_dataset:
            raise ValueError("Training dataset is empty.")

        # 1. Load Model & Tokenizer
        model_name = self.manifest.base_model
        max_seq_length = self.manifest.compute.context_window

        # QLoRA explicitly uses 4-bit loading
        load_in_4bit = True
        if self.manifest.compute.quantization != "4bit":
            logger.warning(
                f"Manifest specifies quantization={self.manifest.compute.quantization}, "
                "but QLoRA enforces 4bit. Overriding to 4bit."
            )

        logger.info(f"Loading Base Model: {model_name} (load_in_4bit={load_in_4bit})")

        if FastLanguageModel is None:
            raise RuntimeError("Unsloth is required.")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=load_in_4bit,
        )

        # 2. Add LoRA Adapters
        logger.info("Applying LoRA Adapters for QLoRA...")
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
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
        )

        def formatting_prompts_func(examples: Dict[str, List[Any]]) -> List[str]:
            """Formats the examples into the instruction prompt template."""
            instructions = examples["instruction"]
            inputs = examples["input"]
            outputs = examples["output"]
            texts = []
            for instruction, input, output in zip(instructions, inputs, outputs, strict=False):
                instr_val = instruction if instruction is not None else ""
                input_val = input if input is not None else ""
                output_val = output if output is not None else ""

                text = (
                    f"""### Instruction:
{instr_val}

### Input:
{input_val}

### Response:
{output_val}"""
                    + tokenizer.eos_token
                )
                texts.append(text)
            return texts

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=hf_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            dataset_num_proc=2,
            formatting_func=formatting_prompts_func,
            args=training_args,
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
            "strategy": "qlora",
        }


def is_bfloat16_supported() -> bool:
    """Checks if bfloat16 is supported on the current device."""
    try:
        import torch

        return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    except Exception:
        return False
