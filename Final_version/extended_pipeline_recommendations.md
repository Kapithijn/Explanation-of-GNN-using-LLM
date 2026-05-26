# Extended Research Pipeline Overview

This folder contains the extended research pipeline implementation for the project. The pipeline is driven by `main.py`, which orchestrates dataset loading, GNN training, shared artifact extraction, experiment execution, LLM inference, reconstruction tasks, evaluation, and result aggregation.

The pipeline is designed for publication-quality experimentation on homogeneous graph datasets such as Elliptic Bitcoin and DGraph using tabular numeric node features.

---

# Research Goals

The extended pipeline supports multiple research directions:

1. **Baseline GNN-to-LLM reasoning**
   - Can an LLM reproduce or explain GNN predictions?

2. **Raw graph reasoning**
   - Can an LLM reason directly over graph structure and numeric node features without latent GNN embeddings?

3. **Embedding-to-subgraph reconstruction**
   - Can an LLM recover graph topology from GNN embeddings?

4. **Structural probing of GNN embeddings**
   - What local graph information is preserved inside node embeddings?

5. **Ablation studies**
   - Does the embedding provide useful information beyond raw graph context?

---

# Updated End-to-End Flow

```text
1. Load and preprocess datasets
2. Build GNN model bundle
3. Train GNN models
4. Extract shared graph artifacts
5. Run experiment branches
    a. Embedding classification
    b. Raw graph reasoning
    c. 1-hop subgraph reconstruction
    d. Structural baselines
6. Evaluate experiment outputs
7. Aggregate metrics across runs
8. Save plots, tables, and serialized outputs
```

---

# Core Architectural Change

The original pipeline followed a mostly linear structure:

```text
Graph → GNN → Embedding → Prompt → LLM → Evaluation
```

The extended pipeline introduces a shared artifact layer and experiment branches:

```text
Graph
    ↓
GNN
    ↓
Shared Extraction Layer
    ↓
Experiment Branches
    ├── Embedding Classification
    ├── Raw Graph Reasoning
    ├── Subgraph Reconstruction
    ├── Structural Baselines
    └── Future Experiments
```

This makes the pipeline modular, extensible, and suitable for rigorous experimentation.

---

# Updated File Structure

```text
main.py
Data_File.py
GNN_Definition.py
Train.py
Extracion.py
Parallel_Extraction.py
LLM_Module.py

experiments/
    embedding_classification.py
    raw_graph_reasoning.py
    subgraph_reconstruction.py
    baselines.py

evaluation/
    classification_metrics.py
    graph_metrics.py

utils/
    serialization.py
    prompt_templates.py
    graph_helpers.py
    metrics.py
```

---

# Updated Pipeline Stages

# 1. Data Stage

### `Data_File.py`

Responsible for dataset loading and preprocessing.

Supported datasets:
- Elliptic Bitcoin
- DGraph

Graph assumptions:
- homogeneous graphs
- numeric/tabular node features

Key functions:

```python
load_dataset(name, **kwargs)
preprocess(data)
print_data_info(data)
```

Responsibilities:
- load graph structure
- normalize node features
- prepare train/val/test masks
- validate graph consistency
- prepare metadata

---

# 2. Model Definition Stage

### `GNN_Definition.py`

Defines the available GNN architectures.

Supported models:
- GCN
- GAT
- GIN
- GraphSAGE

Key function:

```python
build_model_bundle(config)
```

Important notes:
- embeddings are NOT assumed to be interchangeable across architectures
- all experiments must record which GNN produced the embedding
- embedding dimensionality may vary per model

---

# 3. Training Stage

### `Train.py`

Handles model training and evaluation.

Key functions:

```python
train_epoch(...)
evaluate(...)
train_model(...)
train_all(...)
save_model(...)
load_model(...)
```

Publication-quality requirements:
- fixed train/validation/test splits
- deterministic seeds
- multiple independent runs
- checkpoint saving
- config snapshots
- metric logging

---

# 4. Shared Extraction Stage

### `Extracion.py`

This stage becomes the central artifact generator for all experiments.

The extraction layer should NOT contain experiment-specific logic.

Instead, it generates reusable structured artifacts.

Key functions:

```python
get_prediction(...)
get_explanation(...)
get_embedding(...)
get_subgraph(...)
extract_all(...)
```

---

## Updated Extraction Output Schema

Each extracted record should contain:

```python
{
    "dataset": ...,
    "model": ...,
    "target_node": ...,

    "ground_truth_label": ...,
    "prediction": ...,
    "logits": ...,

    "embedding": ...,
    "embedding_dimension": ...,

    "explanation_mask": ...,

    "k_hop_subgraph": ...,
    "one_hop_neighbors": ...,

    "raw_features": ...,
    "neighbor_feature_table": ...,

    "candidate_set": ...,

    "metadata": ...
}
```

---

# 5. Parallel Extraction Stage

### `Parallel_Extraction.py`

Provides worker-side extraction support.

Key function:

```python
extract_one(...)
```

Responsibilities:
- load model state
- load dataset
- run extraction
- serialize artifacts
- support multiprocessing

---

# 6. Experiment Layer

The original single `run_llm_stage()` is replaced by experiment branches.

Experiments consume extraction artifacts and produce outputs independently.

---

# Experiment A — Embedding Classification

### File

```text
experiments/embedding_classification.py
```

### Goal

Determine whether the LLM can reproduce or reason about GNN predictions.

### Inputs

- embedding
- explanation mask
- subgraph summary

### Outputs

- LLM prediction
- explanation
- confidence

### Metrics

- accuracy
- macro-F1
- balanced accuracy

---

# Experiment B — Raw Graph Reasoning

### File

```text
experiments/raw_graph_reasoning.py
```

### Goal

Determine whether the LLM can reason directly over graph structure and numeric node features without embeddings.

### Inputs

- raw node features
- neighborhood summary
- edge list
- optional explanation mask

### Conditions

```text
1. raw features only
2. raw features + neighbors
3. raw features + edge list
4. raw features + explanation mask
```

### Metrics

- accuracy
- precision
- recall
- F1
- AUROC (optional)

### Key Research Question

```text
Does the embedding contain useful information beyond symbolic graph context?
```

---

# Experiment C — 1-Hop Subgraph Reconstruction

### File

```text
experiments/subgraph_reconstruction.py
```

### Goal

Determine whether the LLM can recover local topology from embeddings.

---

## Reconstruction Framing

The task is framed as:

```text
Constrained Neighbor Selection
```

NOT:
- free-form graph generation

---

## Input

```text
- embedding
- optional node features
- optional explanation mask
- candidate node set
```

---

## Candidate Set Construction

Candidate set:

```text
true neighbors
+ sampled non-neighbors
```

Example:

```text
8 true neighbors
32 sampled negatives
40 total candidates
```

---

## Reconstruction Task

Prompt:

```text
Select which candidate nodes are directly connected to the target node.
```

---

## Outputs

```python
{
    "selected_neighbors": [...],
    "confidence": ...
}
```

---

## Evaluation Metrics

- precision@k
- recall@k
- F1
- Jaccard similarity
- graph edit distance
- neighborhood overlap

---

# Experiment D — Structural Baselines

### File

```text
experiments/baselines.py
```

### Purpose

Provide meaningful baselines for reconstruction experiments.

Required baselines:
- random neighbor selection
- cosine similarity nearest neighbors
- feature-distance heuristic

These baselines are required for publication-quality evaluation.

---

# Future Experiment Expansion

Once 1-hop reconstruction is validated:

```text
1-hop reconstruction
    ↓
2-hop reconstruction
    ↓
3-hop reconstruction
```

Higher-hop experiments should reuse the same framework with expanded candidate sets.

---

# 7. LLM Layer

### `LLM_Module.py`

The LLM layer becomes a shared utility module rather than a single-task module.

---

## Responsibilities

- load HuggingFace models
- format prompts
- serialize graph artifacts
- parse structured outputs
- validate responses

---

## Recommended Prompt Builders

```python
build_classification_prompt(...)
build_raw_reasoning_prompt(...)
build_neighbor_selection_prompt(...)
```

---

## Recommended Output Format

All outputs should use deterministic structured formats.

Example:

```json
{
  "selected_neighbors": [12, 44, 58],
  "confidence": 0.81
}
```

This minimizes:
- parsing ambiguity
- formatting errors
- hallucinated graph structures

---

# 8. Evaluation Layer

The original evaluation stage only compared GNN and LLM predictions.

The extended pipeline introduces experiment-specific evaluators.

---

# Classification Evaluation

### File

```text
evaluation/classification_metrics.py
```

Metrics:
- accuracy
- precision
- recall
- F1
- balanced accuracy
- confusion matrix
- calibration metrics

---

# Reconstruction Evaluation

### File

```text
evaluation/graph_metrics.py
```

Metrics:
- edge precision
- edge recall
- edge F1
- node overlap
- Jaccard similarity
- graph edit distance
- neighborhood overlap

---

# Statistical Rigor Requirements

For publication-quality experimentation:

- use multiple random seeds
- report mean ± std
- use identical target-node splits across experiments
- use paired comparisons where appropriate
- separate prompt-tuning validation from final testing
- record all configs and model checkpoints

---

# Configuration Design

Example configuration:

```yaml
datasets:
  - elliptic
  - dgraph

models:
  - gcn
  - gat
  - graphsage

experiments:
  - embedding_classification
  - raw_graph_reasoning
  - reconstruction_1hop
  - cosine_baseline

reconstruction:
  hops: 1
  candidate_ratio: 4
  include_explanation_mask: true
  include_node_features: true
  output_format: json

evaluation:
  num_runs: 5
  seed_base: 42
```

---

# Updated `main.py` Responsibilities

### `main.py`

Updated orchestration flow:

```python
run_data_stage(...)
run_model_build_stage(...)
run_training_stage(...)
run_extraction_stage(...)
run_experiment_stage(...)
run_evaluation_stage(...)
aggregate_results(...)
```

---

## New Responsibilities

- experiment registry
- experiment scheduling
- metric aggregation
- multi-run management
- structured result exporting

---

# Experiment Registry Design

Recommended structure:

```python
EXPERIMENT_REGISTRY = {
    "embedding_classification": ...,
    "raw_graph_reasoning": ...,
    "reconstruction_1hop": ...,
    "cosine_baseline": ...
}
```

Each experiment implements:

```python
prepare_input(...)
build_prompt(...)
run(...)
evaluate(...)
```

---

# Final Pipeline Summary

The extended pipeline transforms the original linear workflow into a modular research framework capable of:

- GNN-to-LLM reasoning experiments
- raw graph reasoning experiments
- topology reconstruction experiments
- structural probing of embeddings
- controlled ablation studies
- publication-quality evaluation across multiple datasets and architectures

The pipeline is now designed as a reusable experimental platform for graph representation and LLM interaction research.
