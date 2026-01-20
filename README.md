# coreason-model-foundry

**Industrial Automation Engine for Training Specialized "Student Models"**

[![License: Prosperity 3.0](https://img.shields.io/badge/license-Prosperity%203.0-blue)](https://prosperitylicense.com/versions/3.0.0)
[![CI](https://github.com/CoReason-AI/coreason-model-foundry/actions/workflows/ci.yml/badge.svg)](https://github.com/CoReason-AI/coreason-model-foundry/actions/workflows/ci.yml)
[![Code Style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Documentation](https://img.shields.io/badge/docs-requirements-brightgreen)](docs/product_requirements.md)

The **coreason-model-foundry** serves as the "Refinery" in the CoReason AI ecosystem. It is an orchestrator for post-training optimization, designed to select the right mathematical strategy (DoRA, ORPO, QLoRA) for the task, prune data for maximum information density, and distribute the resulting artifacts safely.

It implements a **Select-Prune-Train-Merge-Distribute Loop**, utilizing `unsloth` for accelerated training and `mergekit` for model merging.

## Features

*   **Polymorphic Training Architecture:** Dynamically loads the training kernel based on the goal:
    *   **DoRA:** Logic & Math (via `UnslothSFTTrainer`).
    *   **ORPO:** Alignment & Safety (via `UnslothORPOTrainer`).
    *   **QLoRA:** Memory Efficiency (via 4-bit quantization).
*   **Data Curator:** Maximizes "Information Density" using Semantic Deduplication (`SemDeDup`) to remove 95%+ similar duplicates.
*   **Hardware Safety:** "Fail Fast" mechanism prevents OOM crashes by validating VRAM requirements (e.g., enforces 24GB for full ORPO).
*   **The Alchemist (Merging):** Integrates `mergekit` to combine adapters using the **DARE-TIES** algorithm.
*   **Artifact Distribution:** Automatically pushes trained models to the `coreason-publisher` registry.
*   **GxP Compliance:** Calculates provenance hashes (Lot Numbers) for datasets and manifests.

## Installation

```bash
pip install -r requirements.txt
```

*Note: This library relies on `unsloth` and `torch` (CUDA). Ensure these are installed in your environment suitable for your hardware.*

## Usage

### Python API

```python
from coreason_model_foundry import orchestrate_training

# Run the full training pipeline with a manifest file
orchestrate_training("manifest.yaml")
```

### Example Manifest

```yaml
job_id: "train-prod-2025-01-15"
base_model: "unsloth/llama-3-8b-bnb-4bit"

method_config:
  type: "orpo"
  rank: 64
  strict_hardware_check: true

dataset:
  ref: "synthesis://batch_clinical_reasoning"
  sem_dedup: true

compute:
  batch_size: 4
  grad_accum: 4
```
