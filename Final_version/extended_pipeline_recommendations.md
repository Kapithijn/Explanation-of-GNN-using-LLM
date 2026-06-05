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
    b. Embedding classification with explainer subgraph
    c. Raw graph reasoning
    d. 1-hop subgraph reconstruction
    e. Structural baselines
6. Evaluate experiment outputs
7. Aggregate metrics across runs
8. Save plots, tables, and serialized outputs
```

---

# Experiment → Output Mapping (What appears in `outputs/`)

The experiments needed for the thesis map to the current pipeline experiment keys (from `config["experiments"]`) and the JSON files written to `outputs/` as follows.

| Thesis experiment (concept) | Pipeline experiment key | Raw output JSON | Summary output JSON |
|---|---|---|---|
| **DATA → LLM → class (0/1)** (raw graph) | `raw_graph_reasoning` | `outputs/results_raw_raw_graph_reasoning.json` | `outputs/results_summary_raw_graph_reasoning.json` |
| **DATA → GNN → LLM → class (0/1)** (GNN-assisted classification) | `embedding_classification` | `outputs/results_raw_embedding_classification.json` | `outputs/results_summary_embedding_classification.json` |
| **DATA → GNN + GNNExplainer subgraph → LLM → class (0/1)** | `embedding_classification_explainer_subgraph` *(added)* | `outputs/results_raw_embedding_classification_explainer_subgraph.json` | `outputs/results_summary_embedding_classification_explainer_subgraph.json` |
| **DATA → GNN (embedding) → LLM → reconstruct 1-hop neighbors** | `reconstruction_1hop` | `outputs/results_raw_reconstruction_1hop.json` | `outputs/results_summary_reconstruction_1hop.json` |
| **DATA → GNN (embedding + non-subgraph explanation) → LLM → reconstruct 1-hop neighbors** | `reconstruction_1hop_embed_expl` *(added)* | `outputs/results_raw_reconstruction_1hop_embed_expl.json` | `outputs/results_summary_reconstruction_1hop_embed_expl.json` |
| **DATA → (no GNN info) → LLM → reconstruct 1-hop neighbors** (LLM-only reconstruction baseline) | `reconstruction_1hop_no_gnn` *(added)* | `outputs/results_raw_reconstruction_1hop_no_gnn.json` | `outputs/results_summary_reconstruction_1hop_no_gnn.json` |

Notes:
- `raw_graph_reasoning` currently produces `results_raw_raw_graph_reasoning.json` (double “raw”) because the filename is `results_raw_{experiment}.json`.
- `embedding_classification_explainer_subgraph` is a separate classification condition. It keeps the original embedding-classification condition comparable while testing whether a compact GNNExplainer-derived edge list improves LLM agreement with the GNN.
- Reconstruction experiments write *set-based neighbor metrics* (precision/recall/F1/Jaccard/overlap/edit-distance), not classification accuracy.
- Reconstruction experiments and structural baselines also write explainer-reference metrics with `_explainer.json` suffixes, for example `results_summary_reconstruction_1hop_explainer.json`. In those files, the reference set is the GNNExplainer-selected neighbor set rather than the graph's full true one-hop neighborhood.
- Structural (non-LLM) reconstruction baselines are written as:
        - `baseline_random` → `outputs/results_raw_baseline_random.json`, `outputs/results_summary_baseline_random.json`
            - Definition: randomly select `k` nodes from the candidate set as “predicted neighbors” (a pure chance baseline). `k` is set to the number of true neighbors when available; otherwise a small heuristic is used.
        - `baseline_cosine` → `outputs/results_raw_baseline_cosine.json`, `outputs/results_summary_baseline_cosine.json`
            - Definition: rank candidates by cosine similarity between the target node embedding and each candidate’s feature vector, then return the top-`k`.
            - Practical note: this baseline only works when feature vectors are available for the candidate ids (otherwise those ids are skipped).
        - `baseline_feature` → `outputs/results_raw_baseline_feature.json`, `outputs/results_summary_baseline_feature.json`
            - Definition: rank candidates by Euclidean distance between the target node’s raw feature vector and each candidate’s feature vector, then return the closest `k`.

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
    ├── Embedding Classification with Explainer Subgraph
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
    embedding_classification_explainer_subgraph.py
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
    "explainer_edges": ...,
    "explainer_neighbors": ...,

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

# Experiment A2 — Embedding Classification with Explainer Subgraph

### File

```text
experiments/embedding_classification_explainer_subgraph.py
```

### Pipeline key

```text
embedding_classification_explainer_subgraph
```

### Goal

Determine whether an explicit GNNExplainer-derived subgraph improves LLM agreement with the GNN compared with the original embedding-classification condition.

### Inputs

- embedding
- explanation mask summary
- compact explainer subgraph with node ids and important edges

### Explainer Subgraph Construction

The explainer subgraph is built from GNNExplainer edge scores:

```text
1. Rank edges by explainer importance.
2. Normalize importance scores within each target-node explanation.
3. Include edges with normalized importance >= 0.7.
4. Cap the prompt to the top 5 selected edges.
5. If no edge passes the threshold, keep the highest-scoring edge as a fallback.
```

The prompt therefore contains explicit graph structure such as:

```text
Explainer subgraph (normalized importance >= 0.70; fallback top 1):
Nodes: 199342, 198399
Edges:
 - 199342 -> 198399 (importance=0.8123, normalized=1.0000)
```

### Outputs

- LLM prediction
- raw LLM response
- parsed class label

### Metrics

- accuracy
- precision
- recall
- F1
- parse rate

### Key Research Question

```text
Does adding explicit explainer-selected graph structure improve LLM reproduction of GNN predictions?
```

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
  - embedding_classification_explainer_subgraph
  - raw_graph_reasoning
  - reconstruction_1hop
  - reconstruction_1hop_embed_expl
  - reconstruction_1hop_no_gnn
  - baseline_random
  - baseline_cosine
  - baseline_feature

reconstruction:
  hops: 1
  candidate_ratio: 4
  explainer_top_k: 5
  explainer_min_score: null
  include_explanation_mask: true
  include_node_features: true
  output_format: json

prompt_explainer_subgraph:
  normalized_importance_threshold: 0.7
  top_k: 5
  fallback_top_k: 1

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
    "embedding_classification_explainer_subgraph": ...,
    "raw_graph_reasoning": ...,
    "reconstruction_1hop": ...,
    "reconstruction_1hop_embed_expl": ...,
    "reconstruction_1hop_no_gnn": ...,
    "baseline_random": ...,
    "baseline_cosine": ...,
    "baseline_feature": ...
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
- explainer-subgraph classification experiments
- raw graph reasoning experiments
- topology reconstruction experiments
- structural probing of embeddings
- controlled ablation studies
- publication-quality evaluation across multiple datasets and architectures

The pipeline is now designed as a reusable experimental platform for graph representation and LLM interaction research.
