# coreason-model-foundry

**Industrial Automation Engine for Training Specialized "Student Models"**

[![License: Prosperity 3.0](https://img.shields.io/badge/License-Prosperity%203.0-blue)](https://prosperitylicense.com/versions/3.0.0)
[![CI](https://github.com/CoReason-AI/coreason-model-foundry/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason-model-foundry/actions/workflows/ci.yml)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

The **coreason-model-foundry** serves as the "Refinery" in the CoReason AI ecosystem. It is an orchestrator for post-training optimization, designed to select the right mathematical strategy (DoRA, ORPO, QLoRA) for the task, prune data for maximum information density, and distribute the resulting artifacts safely.

It implements a **Select-Prune-Train-Merge-Distribute Loop**, utilizing `unsloth` for accelerated training and `mergekit` for model merging.

---

## Features

-   **Polymorphic Training:** dynamically selects the best training kernel:
    -   **DoRA:** For Logic & Reasoning tasks.
    -   **ORPO:** For Safety & Alignment tasks (requires Triplet data).
    -   **QLoRA:** For memory-efficient fine-tuning.
-   **Data Curator:** Maximizes information density using Semantic Deduplication (`SemDeDup`) with `all-MiniLM-L6-v2`.
-   **Hardware Awareness:** "Fail Fast" mechanism prevents OOM crashes by pre-validating VRAM requirements (e.g., enforces 24GB for full ORPO).
-   **Automated Merging:** Integrates `mergekit` to combine adapters using the **DARE-TIES** algorithm.
-   **GxP Compliance:** Calculates provenance hashes (Lot Numbers) for datasets and manifests.
-   **Artifact Publishing:** Automatically pushes trained models to the CoReason registry.

## Installation

```bash
pip install -r requirements.txt
```

*Note: This library relies on `unsloth` and `torch` (CUDA). Ensure these are installed in your environment suitable for your hardware.*

## Usage

### Training with the Crucible

Create a training manifest (`manifest.yaml`) and run the orchestrator:

```yaml
# manifest.yaml
job_id: "train-prod-2025-01-15"
base_model: "unsloth/llama-3-8b-bnb-4bit"

method_config:
  type: "orpo"
  rank: 64
  alpha: 16
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
  strict_hardware_check: true

dataset:
  ref: "synthesis://batch_clinical_reasoning"
  sem_dedup: true

compute:
  batch_size: 4
  grad_accum: 4
  context_window: 4096
  quantization: "4bit"
```

**Python Usage:**

```python
from coreason_model_foundry import orchestrate_training

# Run the full training pipeline
orchestrate_training("manifest.yaml")
```

### Using the Strategy Factory Directly

```python
from coreason_model_foundry.main import load_manifest
from coreason_model_foundry.strategies.factory import StrategyFactory

manifest = load_manifest("manifest.yaml")
strategy = StrategyFactory.get_strategy(manifest)

# Validate environment and configuration
strategy.validate()

# Train (requires prepared dataset)
# result = strategy.train(dataset)
```
