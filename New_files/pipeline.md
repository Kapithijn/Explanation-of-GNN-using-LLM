---
Pipeline overview

This folder contains the current pipeline implementation for the project. The pipeline is driven by [main.py](main.py), which wires together the other modules in this folder.

Current end-to-end flow

1. Load and preprocess datasets with [Data_File.py](Data_File.py).
2. Build a GNN model bundle with [GNN_Definition.py](GNN_Definition.py).
3. Train the selected models with [Train.py](Train.py).
4. Extract prediction, explanation mask, embedding, and subgraph information for target nodes with [Extracion.py](Extracion.py) and [Parallel_Extraction.py](Parallel_Extraction.py).
5. Format prompts and run LLM inference with [LLM_Module.py](LLM_Module.py).
6. Compare GNN predictions with LLM predictions and save results with [Evalueation.py](Evalueation.py).

File guide

### [main.py](main.py)

Single entry point for the pipeline. It parses CLI arguments, loads config files, applies overrides, resolves per-run settings, and executes the pipeline stages in order.

Key functions:

- `parse_args()` - parses CLI arguments such as config path, datasets, models, target nodes, skip flags, and output controls.
- `load_config(path)` - loads JSON or YAML config data into a dictionary.
- `default_config()` - defines the built-in baseline configuration used when no config file is supplied.
- `apply_cli_overrides(config, args)` - applies CLI settings on top of config values.
- `run_data_stage(config)` - loads datasets and runs preprocessing.
- `run_model_build_stage(config, datasets)` - constructs the model bundle for each dataset.
- `run_training_stage(config, model_bundle, datasets)` - trains all selected model and dataset combinations.
- `run_extraction_stage(config, model_bundle, datasets)` - performs extraction for the selected target nodes, sequentially or in parallel.
- `run_llm_stage(config, extraction_records)` - turns extracted outputs into prompts and runs LLM inference.
- `run_evaluation_stage(config, extraction_records, llm_outputs)` - compares predictions and writes result files.
- `run_pipeline(config, args)` - executes the full pipeline with stage flags and output handling.
- `main(argv=None)` - CLI entry point.

Important behavior in `main.py`:

- Supports `--skip-data`, `--skip-train`, `--skip-extract`, `--skip-llm`, and `--skip-eval`.
- Supports `--num-runs`, `--run-id`, `--seed-base`, and `--output-dir-template` for repeated runs.
- Supports automatic target-node selection through masks, labeled nodes, all nodes, or an explicit node list.
- Supports sequential extraction or multi-process extraction through `ProcessPoolExecutor`.
- Applies a CPU fallback for very large graphs when requested.

### [Data_File.py](Data_File.py)

Handles dataset loading and preprocessing.

Key functions:

- `load_dataset(name, **kwargs)` - loads a dataset by name.
- `preprocess(data)` - prepares the graph data for model input.
- `print_data_info(data)` - prints dataset statistics for inspection.

### [GNN_Definition.py](GNN_Definition.py)

Defines the available GNN models and returns them as a bundle.

Key functions and classes:

- `build_model_bundle(config)` - creates the model registry for the configured input and output dimensions.
- Model classes such as `GCN`, `GAT`, `GIN`, and `GraphSAGE`.

### [Train.py](Train.py)

Contains the training loop for the selected GNN models.

Key functions:

- `train_epoch(model, data, optimizer, criterion)` - performs one training step.
- `evaluate(model, data)` - evaluates the model on a split.
- `train_model(model, data, config)` - runs a full training loop with optional early stopping.
- `train_all(model_bundle, datasets, config, device=None)` - trains all selected model and dataset combinations.
- `save_model(model, path)` - saves model weights.
- `load_model(model, path)` - loads saved model weights.

### [Extracion.py](Extracion.py)

Performs the actual extraction work for one model, one dataset, and one target node.

Key functions:

- `get_prediction(model, data, target_node)` - returns the GNN prediction for the target node.
- `get_explanation(model, data, target_node)` - runs the explainer and returns mask information.
- `get_embedding(model, data, target_node)` - extracts the target node representation.
- `get_subgraph(data, target_node, num_hops)` - returns the k-hop subgraph around the target node.
- `extract_all(model, data, target_node, num_hops=2)` - returns a structured bundle with prediction, explanation, embedding, and subgraph data.

### [Parallel_Extraction.py](Parallel_Extraction.py)

Provides the worker-side extraction helper used by `main.py` when extraction runs in parallel.

Key function:

- `extract_one(...)` - loads the dataset and model state in a worker process and runs extraction for one target node.

### [LLM_Module.py](LLM_Module.py)

Formats extraction outputs into text and runs HuggingFace LLM inference.

Key functions:

- `format_explanation(explanation_mask)` - turns importance scores into human-readable text.
- `format_embedding(embedding, max_length=None)` - serializes the embedding vector, optionally shortening it.
- `build_prompt(explanation_text, embedding_text, subgraph_text, template)` - builds the final prompt.
- `load_llm(model_name, device)` - loads a tokenizer and causal language model.
- `generate_response(model, tokenizer, prompt, device, **gen_kwargs)` - runs text generation.
- `parse_prediction(response)` - extracts the predicted class from a generated response.
- `get_prediction_for_target(model, tokenizer, prompt, device, **gen_kwargs)` - convenience wrapper for one prompt.
- `run_inference_all(model_names, prompts, device, **gen_kwargs)` - runs inference for all configured LLMs.

### [Evalueation.py](Evalueation.py)

Compares GNN predictions with LLM predictions and saves the outputs.

Key functions:

- `compare_predictions(gnn_pred, llm_pred)` - compares one prediction pair.
- `compute_accuracy(results)` - computes an accuracy score.
- `aggregate_results(all_results)` - groups results by model, dataset, and LLM.
- `save_results(results, path, fmt="json")` - writes results to disk.
- `plot_results(results)` - creates plots from the saved results.

Notes on current limitations

- The current pipeline is still centered on the existing fraud-node classification setup.
- The evaluation stage currently compares GNN and LLM predictions after prompt-based inference.
- A future extension can replace or augment the current evaluation with subgraph-to-subgraph comparison, explainer-vs-LLM analysis, or raw-data-to-LLM experiments.

Configuration focus

- Dataset selection is controlled from `main.py` through config or CLI overrides.
- Model selection is handled by the model bundle returned from `GNN_Definition.py`.
- Prompt content is controlled by the prompt template in the config passed into `run_llm_stage()`.
- Output files are written under the configured output directory, with per-run subdirectories when multiple runs are requested.
