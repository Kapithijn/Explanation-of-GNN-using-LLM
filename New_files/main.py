"""Pipeline entry point (New_files).

This file is meant to be the *single* entry point for the whole pipeline described
in `New_files/pipeline.md`.

The intended flow is:
1) Data loading + preprocessing
2) Model construction (GNN bundle)
3) Training (and optional saving)
4) Extraction for one or more target nodes (prediction + explainer masks + embedding + subgraph)
5) Prompt building + LLM inference
6) Evaluation (GNN prediction vs LLM prediction)

This file wires together the modules in `New_files/` into an end-to-end run.
"""
import torch

import argparse
import json
from pathlib import Path
from Data_File import load_dataset, preprocess, print_data_info
from GNN_Definition import build_model_bundle
from Train import train_all
from LLM_Module import format_explanation, format_embedding, build_prompt, run_inference_all
from Evalueation import aggregate_results, save_results


def parse_args(argv=None):
	"""Parse CLI arguments for running the pipeline."""
	parser = argparse.ArgumentParser(
		prog="pipeline",
		description=(
			"Run the end-to-end GNN → extraction → LLM → evaluation pipeline. "
			"See New_files/pipeline.md for intended behavior."
		),
	)

	parser.add_argument(
		"--config",
		type=str,
		default=None,
		help="Path to a JSON/YAML config file describing datasets, models, LLMs, and hyperparameters.",
	)

	# Simple overrides (optional). If provided, they override config values.
	parser.add_argument("--datasets", nargs="*", default=None, help="Override dataset list (e.g., elliptic dgraphfin).")
	parser.add_argument("--models", nargs="*", default=None, help="Override GNN model list (subset of bundle keys).")
	parser.add_argument("--llms", nargs="*", default=None, help="Override LLM model list (HuggingFace model names/paths).")
	parser.add_argument("--target-nodes", nargs="*", type=int, default=None, help="Override target node ids for extraction.")
	parser.add_argument(
		"--num-target-nodes",
		type=int,
		default=None,
		help=(
			"Automatically select N target nodes (used when --target-nodes is not set). "
			"Selection pool and sampling can be controlled with --target-node-pool and --target-node-sampling."
		),
	)
	parser.add_argument(
		"--target-node-pool",
		choices=["test", "train", "val", "labeled", "all"],
		default=None,
		help="Pool to select target nodes from when using --num-target-nodes.",
	)
	parser.add_argument(
		"--target-node-sampling",
		choices=["random", "first"],
		default=None,
		help="How to select nodes from the pool when using --num-target-nodes.",
	)
	parser.add_argument("--num-hops", type=int, default=None, help="Override k for k-hop subgraph extraction.")

	parser.add_argument("--output-dir", type=str, default=None, help="Override output directory for artifacts/results.")
	parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto", help="Device selection.")
	parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")

	# Stage flags.
	parser.add_argument("--skip-data", action="store_true", help="Skip dataset loading/preprocessing stage.")
	parser.add_argument("--skip-train", action="store_true", help="Skip training stage (expects trained weights to exist).")
	parser.add_argument("--skip-extract", action="store_true", help="Skip extraction stage (expects cached extractions to exist).")
	parser.add_argument("--skip-llm", action="store_true", help="Skip LLM inference stage (expects cached LLM outputs to exist).")
	parser.add_argument("--skip-eval", action="store_true", help="Skip evaluation stage.")

	parser.add_argument("--dry-run", action="store_true", help="Print planned stages and exit without running.")
	return parser.parse_args(argv)


def load_config(path):
	"""Load a JSON or YAML config file into a dict."""
	if path is None:
		return {}

	config_path = Path(path)
	if not config_path.exists():
		raise FileNotFoundError(f"Config file not found: {config_path}")

	suffix = config_path.suffix.lower()
	if suffix == ".json":
		with config_path.open("r", encoding="utf-8") as f:
			return json.load(f)

	if suffix in {".yml", ".yaml"}:
		try:
			import yaml  # type: ignore
		except ImportError as exc:
			raise ImportError(
				"YAML config requires PyYAML. Install with: pip install pyyaml"
			) from exc

		with config_path.open("r", encoding="utf-8") as f:
			data = yaml.safe_load(f)
			return data if isinstance(data, dict) else {}

	raise ValueError(f"Unsupported config extension: {suffix} (use .json or .yaml)")


def default_config():
	"""Return a minimal default config dict."""
	return {
		"output_dir": "outputs",
		"datasets": ["elliptic"],
		"models": ["GCN"],
		"llms": ["Qwen/Qwen2.5-0.5B-Instruct"],
		"target_nodes": [],
		"num_target_nodes": 50,
		"target_node_pool": "test",
		"target_node_sampling": "random",
		"num_hops": 2,
		"prompt": {
			"template": (
				"You are given an explanation of a GNN decision.\n\n"
				"Explanation:\n{explanation}\n\n"
				"Embedding:\n{embedding}\n\n"
				"Subgraph:\n{subgraph}\n\n"
				"What class does this node belong to?"
			),
			"embedding_max_length": None,
		},
		"train": {
			"lr": 0.01,
			"epochs": 200,
			"print_every": 20,
			"patience": None,
		},
		"generation": {
			"max_new_tokens": 64,
		},
	}


def apply_cli_overrides(config, args):
	"""Apply CLI overrides on top of a config dict (in-place)."""
	if args.output_dir is not None:
		config["output_dir"] = args.output_dir
	if args.datasets is not None:
		config["datasets"] = args.datasets
	if args.models is not None:
		config["models"] = args.models
	if args.llms is not None:
		config["llms"] = args.llms
	if args.target_nodes is not None:
		config["target_nodes"] = args.target_nodes
	if args.num_target_nodes is not None:
		config["num_target_nodes"] = int(args.num_target_nodes)
		if args.target_nodes is None:
			config["target_nodes"] = []
	if args.target_node_pool is not None:
		config["target_node_pool"] = str(args.target_node_pool)
	if args.target_node_sampling is not None:
		config["target_node_sampling"] = str(args.target_node_sampling)
	if args.num_hops is not None:
		config["num_hops"] = int(args.num_hops)
	if args.seed is not None:
		config["seed"] = int(args.seed)
	if args.device is not None:
		config["device"] = args.device
	return config


def _candidate_target_nodes(data, pool_name: str):
	"""Return a 1D tensor of candidate node indices based on a pool selection."""
	pool = (pool_name or "").strip().lower() or "test"
	if pool in {"train", "val", "test"}:
		mask = getattr(data, f"{pool}_mask", None)
		if mask is None:
			raise ValueError(f"Dataset has no {pool}_mask; cannot sample from pool '{pool}'.")
		return mask.nonzero(as_tuple=False).view(-1)

	if pool == "labeled":
		y = getattr(data, "y", None)
		if y is None:
			return torch.arange(int(getattr(data, "num_nodes", 0)))
		return (y >= 0).nonzero(as_tuple=False).view(-1)

	if pool == "all":
		return torch.arange(int(getattr(data, "num_nodes", 0)))

	raise ValueError(f"Unknown target_node_pool: {pool_name} (expected test/train/val/labeled/all)")


def _select_target_nodes(config, data):
	"""Resolve target nodes from config (explicit list or auto-selection)."""
	explicit = config.get("target_nodes")
	if isinstance(explicit, (list, tuple)) and len(explicit) > 0:
		return [int(v) for v in explicit]

	count = config.get("num_target_nodes")
	if count is None:
		return []
	count = int(count)
	if count <= 0:
		return []

	pool = str(config.get("target_node_pool", "test"))
	sampling = str(config.get("target_node_sampling", "random")).strip().lower() or "random"

	candidates = _candidate_target_nodes(data, pool)
	candidates = candidates.detach().cpu()

	# Ensure candidates are labeled when labels exist.
	y = getattr(data, "y", None)
	if y is not None:
		y = y.detach().cpu()
		if y.numel() == int(getattr(data, "num_nodes", y.numel())):
			candidates = candidates[y[candidates] >= 0]

	if candidates.numel() == 0:
		raise ValueError(f"No candidate target nodes found (pool='{pool}').")

	if count >= int(candidates.numel()):
		selected = candidates
	else:
		if sampling == "first":
			selected = candidates[:count]
		else:
			seed = config.get("seed")
			if seed is None:
				perm = torch.randperm(int(candidates.numel()))
			else:
				generator = torch.Generator(device="cpu")
				generator.manual_seed(int(seed))
				perm = torch.randperm(int(candidates.numel()), generator=generator)
			selected = candidates[perm[:count]]

	return [int(v.item()) for v in selected]


def resolve_device(device):
	"""Resolve a device string (auto → cuda/mps/cpu)."""
	if device is None or device == "auto":
		try:
			import torch
		except Exception:
			return "cpu"

		if torch.cuda.is_available():
			return "cuda"
		if torch.backends.mps.is_available():
			return "mps"
		return "cpu"
	return device


def run_data_stage(config):
	"""Load and preprocess datasets specified in the config."""
	dataset_names = config.get("datasets", [])
	dataset_kwargs = config.get("dataset_kwargs", {})
	should_print = bool(config.get("print_data_info", True))

	datasets = {}
	for dataset_name in dataset_names:
		kwargs = {}
		if isinstance(dataset_kwargs, dict):
			kwargs = dataset_kwargs.get(dataset_name, {}) or {}

		data = load_dataset(dataset_name, **kwargs)
		data = preprocess(data)
		if should_print:
			print_data_info(data)
		datasets[dataset_name] = data

	return datasets


def run_model_build_stage(config, datasets):
	"""Build a model bundle and optionally select a subset by name."""

	if not datasets:
		raise ValueError("No datasets provided. Run the data stage first.")

	example_data = next(iter(datasets.values()))
	if config.get("in_channels") is None:
		config["in_channels"] = int(getattr(example_data, "num_node_features", 0))
	if config.get("out_channels") is None:
		labels = getattr(example_data, "y", None)
		if labels is None:
			raise ValueError("Cannot infer out_channels because dataset has no 'y' labels.")
		try:
			import torch
			valid = labels[labels >= 0]
			if valid.numel() == 0:
				raise ValueError("Cannot infer out_channels because all labels are negative/unlabeled.")
			config["out_channels"] = int(valid.max().item() + 1)
		except Exception:
			config["out_channels"] = int(labels.max().item() + 1)

	bundle = build_model_bundle(config)

	selected_names = config.get("models")
	if not selected_names:
		return bundle

	selected = {}
	for model_name in selected_names:
		if model_name not in bundle:
			available = ", ".join(bundle.keys())
			raise KeyError(f"Unknown model '{model_name}'. Available: {available}")
		selected[model_name] = bundle[model_name]

	return selected


def run_training_stage(config, model_bundle, datasets):
	"""Train all model–dataset combinations and return training histories."""

	train_cfg = config.get("train", {})
	histories = train_all(model_bundle, datasets, train_cfg)
	return {"histories": histories}


def run_extraction_stage(config, model_bundle, datasets):
	"""Run extraction (prediction/explanation/embedding/subgraph) for target nodes."""
	from Extracion import extract_all

	num_hops = int(config.get("num_hops", 2))

	records = []
	for dataset_name, data in datasets.items():
		target_nodes = _select_target_nodes(config, data)
		if not target_nodes:
			raise ValueError(
				"No target nodes specified. Set config['target_nodes'], pass --target-nodes, "
				"or use --num-target-nodes with an appropriate pool (e.g., --target-node-pool test)."
			)
		for model_name, model in model_bundle.items():
			for node_id in target_nodes:
				bundle = extract_all(model, data, node_id, num_hops=num_hops)
				records.append(
					{
						"dataset": dataset_name,
						"model": model_name,
						"target_node": int(node_id),
						"bundle": bundle,
					}
				)

	return records


def _format_subgraph_text(subgraph):
	"""Convert an extracted subgraph (often a dict) into readable text."""
	if isinstance(subgraph, dict):
		num_nodes = subgraph.get("num_nodes", "unknown")
		num_edges = subgraph.get("num_edges", "unknown")
		num_hops = subgraph.get("num_hops", "unknown")
		return f"Subgraph (k={num_hops}) with {num_nodes} nodes and {num_edges} edges."
	return str(subgraph)


def run_llm_stage(config, extraction_records):
	"""Build prompts from extraction records and run LLM inference."""

	llm_names = config.get("llms", [])
	if not llm_names:
		raise ValueError("No LLMs specified. Set config['llms'] or pass --llms, or use --skip-llm.")

	prompt_cfg = config.get("prompt", {})
	template = prompt_cfg.get("template", "{explanation}\n{embedding}\n{subgraph}")
	embedding_max_length = prompt_cfg.get("embedding_max_length")

	prompts = []
	for record in extraction_records:
		bundle = record["bundle"]
		edge_mask = bundle.get("explanation", {}).get("edge_mask")
		embedding = bundle.get("embedding", {}).get("embedding")
		subgraph = bundle.get("subgraph")

		if edge_mask is None:
			explanation_text = "No explanation edge mask available."
		else:
			explanation_text = format_explanation(torch.tensor(edge_mask))

		embedding_text = format_embedding(embedding, max_length=embedding_max_length)
		subgraph_text = _format_subgraph_text(subgraph)

		prompt = build_prompt(explanation_text, embedding_text, subgraph_text, template)
		prompts.append(prompt)

	device = resolve_device(config.get("device"))

	generation_cfg = config.get("generation", {}) or {}
	if not isinstance(generation_cfg, dict):
		generation_cfg = {}

	predictions = run_inference_all(llm_names, prompts, device, **generation_cfg)

	return {"prompts": prompts, "predictions": predictions}


def run_evaluation_stage(config, extraction_records, llm_outputs):
	"""Compare GNN predictions vs LLM predictions and save results."""

	prompts = llm_outputs.get("prompts", [])
	predictions_by_llm = llm_outputs.get("predictions", {})

	comparisons = []
	for llm_name, llm_preds in predictions_by_llm.items():
		if len(llm_preds) != len(extraction_records):
			raise ValueError(
				f"LLM '{llm_name}' returned {len(llm_preds)} predictions, "
				f"but there are {len(extraction_records)} extraction records."
			)

		for idx, record in enumerate(extraction_records):
			gnn_pred = record["bundle"]["prediction"]["predicted_class"]
			llm_pred = llm_preds[idx]
			comparisons.append(
				{
					"dataset": record["dataset"],
					"model": record["model"],
					"llm": llm_name,
					"target_node": record["target_node"],
					"gnn_pred": gnn_pred,
					"llm_pred": llm_pred,
					"prompt": prompts[idx] if idx < len(prompts) else None,
				}
			)

	grouped = {}
	for row in comparisons:
		group = f"{row['model']}|{row['dataset']}|{row['llm']}"
		grouped.setdefault(group, []).append({"gnn_pred": row["gnn_pred"], "llm_pred": row["llm_pred"]})

	summary = aggregate_results(grouped)

	output_dir = Path(str(config.get("output_dir", "outputs")))
	output_dir.mkdir(parents=True, exist_ok=True)
	results_summary_path = str(output_dir / "results_summary.json")
	results_raw_path = str(output_dir / "results_raw.json")
	save_results(summary, results_summary_path, fmt="json")
	save_results(comparisons, results_raw_path, fmt="json")

	return {
		"summary": summary,
		"comparisons": comparisons,
		"paths": {"summary": results_summary_path, "raw": results_raw_path},
	}


def run_pipeline(config, args):
	"""Execute the full pipeline end-to-end in order."""
	merged = default_config()
	merged.update(config)
	apply_cli_overrides(merged, args)

	device = resolve_device(merged.get("device"))
	merged["device"] = device

	stages = {
		"data": not args.skip_data,
		"train": not args.skip_train,
		"extract": not args.skip_extract,
		"llm": not args.skip_llm,
		"eval": not args.skip_eval,
	}

	if args.dry_run:
		print("Planned stages:")
		for name, enabled in stages.items():
			print(f"- {name}: {'ON' if enabled else 'OFF'}")
		print(f"Device: {device}")
		print(f"Output dir: {merged.get('output_dir')}")
		return {"config": merged, "stages": stages}

	output_dir = Path(str(merged.get("output_dir", "outputs")))
	output_dir.mkdir(parents=True, exist_ok=True)

	if merged.get("seed") is not None:
		from Train import set_seed
		set_seed(int(merged["seed"]))

	state = {"config": merged, "stages": stages}

	datasets = None
	model_bundle = None
	extraction_records = None
	llm_outputs = None

	if stages["data"]:
		datasets = run_data_stage(merged)
		state["datasets"] = datasets
	else:
		state["datasets"] = None

	if stages["train"]:
		if datasets is None:
			raise RuntimeError("Training requested but datasets are missing. Run data stage or implement dataset caching.")
		model_bundle = run_model_build_stage(merged, datasets)
		state["model_bundle"] = list(model_bundle.keys())

		training_artifacts = run_training_stage(merged, model_bundle, datasets)
		state["training"] = training_artifacts
	else:
		state["training"] = None

	if stages["extract"]:
		if datasets is None:
			raise RuntimeError("Extraction requested but datasets are missing.")
		if model_bundle is None:
			raise RuntimeError("Extraction requested but models are missing. Train or load models first.")
		extraction_records = run_extraction_stage(merged, model_bundle, datasets)
		state["extractions"] = extraction_records
	else:
		state["extractions"] = None

	if stages["llm"]:
		if extraction_records is None:
			raise RuntimeError("LLM inference requested but extraction records are missing.")
		llm_outputs = run_llm_stage(merged, extraction_records)
		state["llm"] = llm_outputs
	else:
		state["llm"] = None

	if stages["eval"]:
		if extraction_records is None:
			raise RuntimeError("Evaluation requested but extraction records are missing.")
		if llm_outputs is None:
			raise RuntimeError("Evaluation requested but LLM outputs are missing.")
		evaluation = run_evaluation_stage(merged, extraction_records, llm_outputs)
		state["evaluation"] = evaluation
	else:
		state["evaluation"] = None

	return state


def main(argv=None):
	"""CLI entry point."""
	args = parse_args(argv)
	config = load_config(args.config)
	state = run_pipeline(config, args)

	evaluation = state.get("evaluation")
	if isinstance(evaluation, dict):
		paths = evaluation.get("paths")
		if isinstance(paths, dict):
			summary_path = paths.get("summary")
			raw_path = paths.get("raw")
			if summary_path or raw_path:
				print("Saved results:")
				if summary_path:
					print(f"- summary: {summary_path}")
				if raw_path:
					print(f"- raw: {raw_path}")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())
