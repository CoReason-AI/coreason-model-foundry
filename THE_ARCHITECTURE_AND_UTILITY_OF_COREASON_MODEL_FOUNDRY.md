# The Architecture and Utility of coreason-model-foundry

### 1. The Philosophy (The Why)

In the modern MLOps landscape, "training a model" is no longer a monolithic task—it is a nuanced orchestration of mathematics, hardware constraints, and data hygiene. Generic training scripts often fail to capture this complexity, leaving engineers to cobble together fragmented tools for quantization, safety alignment, and artifact management.

**coreason-model-foundry** was built to replace ad-hoc scripts with a **Polymorphic Training Architecture**. It operates on the principle of "The Right Math for the Right Task." Instead of a one-size-fits-all trainer, the Foundry acts as a dynamic kernel selector: it deploys **DoRA** (Weight-Decomposed Low-Rank Adaptation) for logic-heavy tasks, **ORPO** (Odds Ratio Preference Optimization) for safety alignment, and **QLoRA** for memory-constrained environments.

Crucially, the Foundry extends the definition of training beyond gradient descent. It implements a strict **Select-Prune-Train-Merge-Distribute** lifecycle. By integrating semantic deduplication (`SemDeDup`) before training and automated merging (**DARE-TIES**) after, it ensures that models are not just trained, but refined and synthesized into deployment-ready artifacts. It is not just a trainer; it is an industrial refinery for Student Models.

### 2. Under the Hood (The Dependencies & Logic)

The package leverages a potent stack of specialized libraries to achieve speed and precision:

*   **Unsloth:** The engine's core. By utilizing `unsloth`'s hand-written Triton kernels, the Foundry bypasses standard PyTorch overhead, enabling 2x faster training and 4-bit quantization. This allows Llama-3 class models to be fine-tuned on consumer-grade hardware without sacrificing fidelity.
*   **Mergekit:** The synthesis layer. The Foundry wraps `mergekit` to execute **DARE-TIES** (Drop And REscale with Task Arithmetic) algorithms. This allows distinct adapter weights (e.g., a "Reasoning" adapter and a "Safety" adapter) to be mathematically merged into a single, cohesive model.
*   **Sentence-Transformers:** The curator's lens. Before a single gradient is updated, the **SemDeDup** module uses `all-MiniLM-L6-v2` to embed the training data. It clusters semantically identical records and aggressively prunes redundancy (defaulting to 95% similarity), ensuring the model learns from unique signals rather than rote repetition ("Information Density").
*   **Pydantic & Veritas:** The governance layer. `pydantic` strictly validates the **TrainingManifest**, enforcing hardware constraints (e.g., failing fast if ORPO is requested on <24GB VRAM). `coreason-veritas` acts as the auditor, hashing datasets and logging loss metrics to ensure GxP-compliant traceability.

### 3. In Practice (The How)

The Foundry abstracts the complexity of kernel selection and hardware management behind a clean, strategy-driven API.

#### Example 1: The Strategy Recipe
The entry point is the `TrainingManifest`. This declarative configuration allows you to define *what* you want to achieve, while the Foundry figures out *how* to execute it on your available hardware.

```python
from coreason_model_foundry.schemas import TrainingManifest, MethodConfig, MethodType

# Define a "Safety Alignment" mission using ORPO
manifest = TrainingManifest(
    job_id="safety-v2-prod",
    base_model="unsloth/llama-3-8b-bnb-4bit",
    method_config=MethodConfig(
        type=MethodType.ORPO,  # Selects the Odds Ratio Preference Optimizer
        rank=64,
        strict_hardware_check=True  # Will raise Error if VRAM < 24GB
    ),
    compute={
        "batch_size": 4,
        "grad_accum": 4
    }
)
```

#### Example 2: The Execution Loop
The `StrategyFactory` reads the manifest and instantiates the correct trainer (e.g., `ORPOStrategy` vs `DoRAStrategy`). The lifecycle handles data pruning, training, and artifact handoff automatically.

```python
from coreason_model_foundry.strategies.factory import StrategyFactory
from coreason_model_foundry.curator import DataCurator

# 1. Select the Strategy (Polymorphism in action)
strategy = StrategyFactory.get_strategy(manifest)

# 2. Prune Data (SemDeDup)
# Loads data, embeds it, and removes 95% similar duplicates
curator = DataCurator()
train_dataset = curator.prepare_dataset(
    source="synthesis://safety-data-v1",
    deduplicate=True
)

# 3. Train & Distribute
# - Checks hardware
# - Optimizes with Unsloth kernels
# - Pushes final artifact to coreason-publisher
strategy.train(train_dataset)
```

#### Example 3: The Alchemist (Merging Adapters)
Once models are trained, the **Alchemist** can fuse them using Task Arithmetic, creating a "Super-Model" that retains the capabilities of both parents.

```python
from coreason_model_foundry.alchemist import Alchemist
from coreason_model_foundry.schemas import MergeRecipe

# Combine a Logic Adapter and a Safety Adapter
recipe = MergeRecipe(
    base_model="unsloth/llama-3-8b-bnb-4bit",
    adapters=[
        {"path": "models/logic-adapter", "weight": 1.0, "density": 0.5},
        {"path": "models/safety-adapter", "weight": 0.8, "density": 0.7}
    ],
    method="dare_ties"
)

# Execute the merge via CLI wrapper
alchemist = Alchemist()
final_model_path = alchemist.merge(recipe)
print(f"Fused model ready at: {final_model_path}")
```
