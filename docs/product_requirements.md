# **Product Requirements Document: coreason-model-foundry**

Domain: MLOps, Knowledge Distillation, & Post-Training Optimization
Architectural Role: The "Refinery" / The Fine-Tuning Orchestrator
Core Philosophy: "The Right Math for the Right Task. Train fast (Unsloth), Merge often (DARE), Distribute safely."
Dependencies: coreason-synthesis (Data), coreason-veritas (Audit), coreason-publisher (Artifacts), unsloth (Optimization), mergekit (Merging)

## **---**

**1. Executive Summary**

coreason-model-foundry is the industrial automation engine for training specialized "Student Models."

It utilizes a **Polymorphic Training Architecture** optimized by **Unsloth**. It allows SREs to select the exact mathematical strategy (DoRA for logic, ORPO for safety, QLoRA for efficiency) and executes it 2x faster than standard HuggingFace trainers. Crucially, it includes an **MLOps Handoff** layer, ensuring that once a model is trained and verified, it is automatically pushed to the coreason-publisher artifact registry for deployment.

## **2. Functional Philosophy**

The agent must implement the **Select-Prune-Train-Merge-Distribute Loop**:

1. **Strategy Selection:** Dynamically loads the training kernel (dora, orpo, qlora) based on the goal.
2. **Semantic Pruning (SemDeDup):** Embeds input data and removes 95%+ similar duplicates to maximize "Information Density."
3. **Unsloth Acceleration:** Uses FastLanguageModel kernels to bypass PyTorch overhead, enabling Llama-3 fine-tuning on consumer-grade hardware.
4. **Post-Training Merging:** Uses **DARE-TIES** to combine adapters (e.g., Safety + Logic) into a single artifact.
5. **Artifact Distribution:** Models are not left on disk. The Foundry pushes the final binary (adapter.bin) to coreason-publisher for Git LFS versioning.

## **---**

**3. Core Functional Requirements (Component Level)**

### **3.1 The Method Registry (The Strategy Engine)**

**Concept:** A Factory Pattern that instantiates the correct Trainer.

* **Strategies:**
  * **dora:** UnslothSFTTrainer(use_dora=True). Best for Logic/Math.
  * **orpo:** UnslothORPOTrainer. Best for Alignment/Safety.
  * **qlora:** UnslothSFTTrainer(load_in_4bit=True). Best for Memory Efficiency.
* **Hardware Safety (Restored):**
  * **Constraint:** Before loading, check torch.cuda.get_device_properties(0).total_memory.
  * *Logic:* If Method=ORPO AND VRAM < 24GB: **Fail Fast** with HardwareIncompatibleError (unless allow_gradient_checkpointing is forced). Prevent OOM crashes before they start.

### **3.2 The Data Curator (The Pruner)**

**Concept:** Maximizes "Information Density."

* **SemDeDup Module:**
  * *Embed:* Uses all-MiniLM-L6-v2.
  * *Cluster:* Cosine Similarity > 0.95.
  * *Prune:* Keeps top 1 representative per cluster.
* **Format Adapter:**
  * Standardizes inputs (Triplets for ORPO, Instructions for SFT) into the Unsloth-compatible format.

### **3.3 The Crucible (The Execution Engine)**

**Concept:** The unsloth wrapper.

* **Optimization:** Loads FastLanguageModel with 4-bit quantization.
* **Target Modules:** Auto-targets all linear layers (q,k,v,o,gate,up,down) for maximum plasticity.
* **Logging:** Streams loss/epoch to coreason-veritas every 10 steps.
* **Output:** Saves Adapter weights and Tokenizer.

### **3.4 The Alchemist (The Merger)**

**Concept:** mergekit wrapper.

* **Algorithm:** **DARE-TIES** (Drop And REscale with Task Arithmetic).
* **Action:** Merges disparate adapters into a unified model directory.

### **3.5 The Examiner (The Quality Gate)**

**Concept:** Automated verification.

* **Decontamination:** checks for N-gram overlap between Train and Test sets.
* **Thresholding:** Fails the build if specific metrics (Safety Rate, Logic Score) do not beat the Baseline.

## **---**

**4. Integration Requirements**

* **coreason-synthesis:** Source of raw training data.
* **coreason-veritas:** Audit logging (hashes of data and manifests).
* **coreason-publisher (Restored):**
  * Foundry does *not* deploy models.
  * Upon success, Foundry calls publisher.push_artifact(path, version_tag).
  * Publisher handles the Git LFS / S3 upload and notifies coreason-cortex of availability.
* **coreason-search:** Used for embedding during SemDeDup.

## **---**

**5. User Stories**

### **Story A: The "Logical" Upgrade (DoRA)**

Config: method: dora.
Action: Unsloth trains a reasoning adapter 2x faster.
Result: Logic score improves. Model pushed to Publisher.

### **Story B: The "Safety" Patch (ORPO)**

Config: method: orpo with Refusal Triplets.
Safety Check: System detects 16GB VRAM. ORPO requires 24GB.
Action: Fail Fast. Error: "Insufficient VRAM for ORPO. Use Gradient Checkpointing or upgrade node."
Result: SRE avoids a wasted 3-hour crash.

### **Story C: The "Super-Model" (Merging)**

Config: Merge Logic_Adapter + Safety_Adapter.
Action: Alchemist combines weights using DARE-TIES.
Result: A single artifact containing both behaviors is created and published.

## **---**

**6. Data Schema**

### **TrainingManifest (YAML)**

```yaml
job_id: "train-prod-2025-01-15"
base_model: "unsloth/llama-3-8b-bnb-4bit"

# STRATEGY
method_config:
  type: "orpo"
  rank: 64
  strict_hardware_check: true # Fail if VRAM is low

# PUBLISHING (Restored)
publish_target:
  registry: "s3://coreason-models/prod"
  tag: "v2.1.0-safety"
  trigger_deployment: false

# INFRASTRUCTURE
compute:
  batch_size: 4
  grad_accum: 4
```

## **---**

**7. Implementation Directives for the Coding Agent**

1. **Unsloth Mandatory:** Use unsloth for all training logic. Fallback to peft only if absolutely necessary (and log a warning).
2. **Publisher Handoff:** The train() function must end with a call to coreason_publisher.upload(). Do not leave files stranding on the GPU node.
3. **Pre-Flight Checks:** Implement a check_hardware() function that runs *before* loading the model. Compare request methodology against torch.cuda.mem_get_info().
4. **Hashing:** Calculate SHA256 of the dataset and manifest before training. This is the "Lot Number" for GxP compliance.
