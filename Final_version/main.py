"""Pipeline MAin.


The intended pipeline is:
1) Data loading + preprocessing
2) Model construction (GNN bundle)
3) Training (and optional saving)
4) Extraction for one or more target nodes (prediction + explainer masks + embedding + subgraph)
5) Prompt building + LLM inference
6) Evaluation (GNN prediction vs LLM prediction)

This file wires together the modules in `New_files/` into an end-to-end run.
"""
import torch

import copy

import argparse
import json
from pathlib import Path
from Final_version.Data_File import load_dataset, preprocess, print_data_info
from Final_version.GNN_Definition import build_model_bundle
from Final_version.Train import train_all
from Final_version.LLM_Module import (
	format_explanation,
	format_embedding,
	build_prompt,
	build_classification_prompt,
	build_raw_reasoning_prompt,
	build_neighbor_selection_prompt,
	run_inference_all,
	parse_neighbor_selection_response,
)
from Final_version.Evalueation import aggregate_results, save_results, compute_classification_metrics, evaluate_reconstruction
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from Final_version.Parallel_Extraction import extract_one

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
	parser.add_argument(
		"--num-runs",
		type=int,
		default=1,
		help="Number of full pipeline runs (use with --seed-base for variance sweeps).",
	)
	parser.add_argument(
		"--run-id",
		type=str,
		default=None,
		help="Optional run identifier (used in output dir templates).",
	)
	parser.add_argument(
		"--seed-base",
		type=int,
		default=None,
		help="Base seed for multi-run; run i uses seed_base + i.",
	)
	parser.add_argument(
		"--output-dir-template",
		type=str,
		default=None,
		help="Template for output dir; supports {base} and {run_id}.",
	)
	parser.add_argument(
		"--extract-workers",
		type=int,
		default=None,
		help="Number of worker processes for extraction (1 = sequential).",
	)
	parser.set_defaults(large_graph_cpu_fallback=None)
	fallback_group = parser.add_mutually_exclusive_group()
	fallback_group.add_argument(
		"--large-graph-cpu-fallback",
		action="store_true",
		dest="large_graph_cpu_fallback",
		help="Enable automatic CPU fallback for very large graphs and CUDA OOM retries.",
	)
	fallback_group.add_argument(
		"--no-large-graph-cpu-fallback",
		action="store_false",
		dest="large_graph_cpu_fallback",
		help="Disable automatic CPU fallback for very large graphs and CUDA OOM retries.",
	)

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
		"datasets": ["elliptic"],#elliptic
		"models": ["GAT"],  # subset of bundle keys or None for all
		"llms": ["Qwen/Qwen2.5-3B-Instruct"],#Qwen/Qwen3.5-2B or Qwen/Qwen2.5-0.5B-Instruct Qwen2.5-3B-Instruct
		"experiments": [
			"embedding_classification",
			"raw_graph_reasoning",
			"reconstruction_1hop",
			"baseline_random",
			"baseline_cosine",
			"baseline_feature",
		],
		"extract_workers": 1,
		"target_nodes": [],	
		"num_target_nodes": 1,
		"target_node_pool": "test",
		"target_node_sampling": "random",
		"num_hops": 2, 
		"raw_graph_reasoning": {
			"condition": "raw_features_neighbors",
		},
		"reconstruction": {
			"candidate_ratio": 4,
			"include_explanation_mask": False,
			"include_node_features": False,
			"output_format": "json",
		},
		"prompt": {
			"template": (
"You are helping compliance analysts understand and validate the prediction of a graph neural network (GNN) used for financial transaction fraud detection on a transaction graph.\n\n"
"You will see examples of GNN decisions with their correct class, then a new case to classify.\n"
"The explanation lists \"Index <id> with importance <score>\"; higher importance means more influence on the decision.\n\n"
"Class definition:\n"
"- 0 = licit (normal) transaction\n"
"- 1 = illicit (suspicious) transaction\n\n"
"Example 1\n"
"Explanation:\n"
"Top important edges/features:\n"
" - Index 172445 with importance 0.6319\n"
" - Index 156241 with importance 0.0000\n"
" - Index 156227 with importance 0.0000\n"
" - Index 156228 with importance 0.0000\n"
" - Index 156229 with importance 0.0000\n\n"
"Embedding:\n"
"embedding: [3.1249, 5.1401, 0.8050, 1.7747, 0.0000, 2.2467, 0.0000, 0.0000, 3.0109, 0.0000, 5.3093, 0.0000, 0.0000, 1.2692, 0.0000, 2.2743, 0.7417, 0.0000, 0.0000, 1.2539, 0.0000, 0.0000, 2.7624, 0.0000, 0.0000, 0.0000, 0.6356, 2.1562, 0.0000, 0.0000, 0.6110, 0.0000, 1.6118, 0.0000, 3.5684, 0.0000, 0.0000, 0.0000, 0.0000, 1.3218, 0.0000, 0.0000, 0.0000, 2.3571, 0.0000, 0.5580, 0.7710, 2.5565, 0.0000, 0.0000, 0.0000, 0.0511, 2.0591, 0.0480, 3.9548, 0.0000, 2.6845, 0.0000, 0.0000, 1.9812, 0.0000, 0.0000, 0.0000, 0.2535]\n\n"
"Subgraph:\n"
"Subgraph (k=2) with 2 nodes and 1 edges.\n\n"
"Correct label: 1 (illicit)\n\n"
"Example 2\n"
"Explanation:\n"
"Top important edges/features:\n"
" - Index 156242 with importance 0.0000\n"
" - Index 156228 with importance 0.0000\n"
" - Index 156229 with importance 0.0000\n"
" - Index 156230 with importance 0.0000\n"
" - Index 156231 with importance 0.0000\n\n"
"Embedding:\n"
"embedding: [8.4794, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 9.1156, 9.7832, 0.0000, 11.5688, 0.0000, 0.0000, 10.0358, 11.9583, 0.0000, 0.0000, 3.3227, 11.6687, 3.8521, 13.1241, 0.0000, 7.0590, 7.8354, 0.0000, 0.0000, 0.0000, 8.7543, 12.2876, 0.0000, 0.0827, 0.0000, 7.4203, 0.0000, 6.4304, 0.0000, 3.4102, 0.2344, 7.0176, 7.4691, 9.7183, 0.0000, 1.6065, 2.8625, 0.3167, 0.0000, 0.0000, 0.0000, 2.5734, 0.0000, 0.0000, 0.7634, 0.0000, 0.0000, 6.4666, 0.7579, 0.0000, 1.7948, 3.4339, 0.0000, 9.7507, 0.0000, 9.2003]\n\n"
"Subgraph:\n"
"Subgraph (k=2) with 1 nodes and 0 edges.\n\n"
"Correct label: 0 (licit)\n\n"
"Now classify the following case.\n\n"
"Explanation:\n{explanation}\n\n"
"Embedding:\n{embedding}\n\n"
"Subgraph:\n{subgraph}\n\n"
"Return your answer in exactly this format:\n"
"The predicted class is X\n"
"where X is either 0 (licit) or 1 (illicit). Do not output anything else.\n"
			),
			"embedding_max_length": None,
		},
		"prompt_raw_reasoning": {
			"template": (
"You are given a transaction node and optional graph context.\n\n"
"Class definition:\n"
"- 0 = licit (normal) transaction\n"
"- 1 = illicit (suspicious) transaction\n\n"
"Raw features for target node:\n{raw_features}\n\n"
"Neighbor feature table (if provided):\n{neighbor_table}\n\n"
"Edge list (if provided):\n{edge_list}\n\n"
"Return your answer in exactly this format:\n"
"The predicted class is X\n"
"where X is either 0 or 1. Do not output anything else."
			)
		},
		"prompt_reconstruction": {
			"template": (
"You are given a target node embedding and a candidate set of neighbor nodes.\n"
"Select which candidate nodes are directly connected to the target node.\n\n"
"Embedding:\n{embedding}\n\n"
"Candidate set (node ids):\n{candidates}\n\n"
"Return a JSON object exactly in this format:\n"
"{\"selected_neighbors\": [<ids>], \"confidence\": <float>}\n"
"Do not output anything else."
			)
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
		"large_graph_cpu_fallback": True,
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
	if getattr(args, "extract_workers", None) is not None and args.extract_workers is not None:
		config["extract_workers"] = max(1, int(args.extract_workers))
	if getattr(args, "no_large_graph_cpu_fallback", False):
		config["large_graph_cpu_fallback"] = False
	elif getattr(args, "large_graph_cpu_fallback", None):
		config["large_graph_cpu_fallback"] = True
	if args.device is not None:
		config["device"] = args.device
	return config


def _resolve_base_output_dir(config, args):
	"""Resolve the base output directory before applying run-specific suffixes."""
	if args.output_dir is not None:
		return str(args.output_dir)
	if isinstance(config, dict) and config.get("output_dir") is not None:
		return str(config.get("output_dir"))
	return "outputs"


def _resolve_run_id(args, run_index):
	"""Return a run id string or None for the single-run default."""
	num_runs = int(getattr(args, "num_runs", 1) or 1)
	if num_runs <= 1:
		return str(args.run_id) if args.run_id is not None else None
	if args.run_id:
		return f"{args.run_id}_{run_index + 1}"
	return str(run_index + 1)


def _resolve_run_output_dir(config, args, run_id):
	"""Return the per-run output directory path (or None to keep defaults)."""
	output_template = getattr(args, "output_dir_template", None)
	if output_template:
		base_dir = _resolve_base_output_dir(config, args)
		safe_run_id = run_id if run_id is not None else "1"
		return output_template.format(base=base_dir, run_id=safe_run_id)

	num_runs = int(getattr(args, "num_runs", 1) or 1)
	if run_id is None and num_runs <= 1:
		return args.output_dir

	base_dir = _resolve_base_output_dir(config, args)
	if run_id is None:
		return base_dir
	return str(Path(base_dir) / f"run_{run_id}")


def _resolve_run_seed(config, args, run_index):
	"""Return per-run seed or None when not specified."""
	base = getattr(args, "seed_base", None)
	if base is None:
		base = getattr(args, "seed", None)
	if base is None and isinstance(config, dict):
		base = config.get("seed")
	if base is None:
		return None
	return int(base) + int(run_index)


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
	skipped = []
	for dataset_name in dataset_names:
		kwargs = {}
		if isinstance(dataset_kwargs, dict):
			kwargs = dataset_kwargs.get(dataset_name, {}) or {}

		try:
			data = load_dataset(dataset_name, **kwargs)
		except FileNotFoundError as exc:
			skipped.append((dataset_name, str(exc)))
			continue
		data = preprocess(data)
		if should_print:
			print_data_info(data)
		datasets[dataset_name] = data

	if skipped:
		for dataset_name, message in skipped:
			print(f"Skipping dataset '{dataset_name}': {message}")

	if not datasets:
		raise FileNotFoundError("No requested datasets could be loaded. Check that the raw dataset files are present.")

	return datasets


def _build_model_config_for_data(config, data):
	"""Infer model dimensions from a specific dataset."""
	model_config = dict(config)
	if model_config.get("in_channels") is None:
		model_config["in_channels"] = int(getattr(data, "num_node_features", 0))
	if model_config.get("out_channels") is None:
		labels = getattr(data, "y", None)
		if labels is None:
			raise ValueError("Cannot infer out_channels because dataset has no 'y' labels.")
		try:
			import torch
			valid = labels[labels >= 0]
			if valid.numel() == 0:
				raise ValueError("Cannot infer out_channels because all labels are negative/unlabeled.")
			model_config["out_channels"] = int(valid.max().item() + 1)
		except Exception:
			model_config["out_channels"] = int(labels.max().item() + 1)
	return model_config


def _resolve_runtime_device_for_data(requested_device, data, enable_large_graph_cpu_fallback=True):
	"""Prefer CPU for very large graphs when CUDA is requested."""
	device = resolve_device(requested_device)
	if device != "cuda" or not enable_large_graph_cpu_fallback:
		return device

	num_nodes = int(getattr(data, "num_nodes", 0) or 0)
	num_edges = int(data.edge_index.size(1)) if getattr(data, "edge_index", None) is not None else 0
	if num_nodes >= 1_000_000 or num_edges >= 2_000_000:
		return "cpu"
	return device


def run_model_build_stage(config, datasets):
	"""Build one model bundle per dataset and optionally select a subset by name."""

	if not datasets:
		raise ValueError("No datasets provided. Run the data stage first.")

	selected_names = config.get("models")
	bundles = {}
	for dataset_name, data in datasets.items():
		model_config = _build_model_config_for_data(config, data)
		bundle = build_model_bundle(model_config)

		if selected_names:
			selected = {}
			for model_name in selected_names:
				if model_name not in bundle:
					available = ", ".join(bundle.keys())
					raise KeyError(f"Unknown model '{model_name}'. Available: {available}")
				selected[model_name] = bundle[model_name]
			bundle = selected

		bundles[dataset_name] = bundle

	return bundles


def run_training_stage(config, model_bundle, datasets):
	"""Train all model–dataset combinations and return training histories."""

	train_cfg = config.get("train", {})
	histories = {}
	enable_large_graph_cpu_fallback = bool(config.get("large_graph_cpu_fallback", True))
	for dataset_name, data in datasets.items():
		bundle = model_bundle.get(dataset_name)
		if bundle is None:
			continue
		device = _resolve_runtime_device_for_data(
			config.get("device"),
			data,
			enable_large_graph_cpu_fallback=enable_large_graph_cpu_fallback,
		)
		per_dataset_histories = train_all(bundle, {dataset_name: data}, train_cfg, device=device)
		histories[dataset_name] = per_dataset_histories
	return {"histories": histories}


def run_extraction_stage(config, model_bundle, datasets):
	"""Run extraction (prediction/explanation/embedding/subgraph) for target nodes."""
	from Final_version.Extracion import extract_all, build_candidate_set, get_one_hop_neighbors

	num_hops = int(config.get("num_hops", 2))
	enable_large_graph_cpu_fallback = bool(config.get("large_graph_cpu_fallback", True))

	try:
		from tqdm.auto import tqdm  # type: ignore
	except Exception:
		tqdm = None

	# Resolve and freeze the target node lists once per dataset (important when sampling is random).
	target_nodes_by_dataset = {}
	total = 0
	for dataset_name, data in datasets.items():
		if dataset_name not in model_bundle:
			continue
		device = _resolve_runtime_device_for_data(
			config.get("device"),
			data,
			enable_large_graph_cpu_fallback=enable_large_graph_cpu_fallback,
		)
		target_nodes = _select_target_nodes(config, data)
		if not target_nodes:
			raise ValueError(
				"No target nodes specified. Set config['target_nodes'], pass --target-nodes, "
				"or use --num-target-nodes with an appropriate pool (e.g., --target-node-pool test)."
			)
		target_nodes_by_dataset[dataset_name] = [int(v) for v in target_nodes]
		total += len(target_nodes_by_dataset[dataset_name]) * len(model_bundle[dataset_name])

	if total == 0:
		return []

	print(
		f"Starting extraction for {total} node(s) "
		f"({len(target_nodes_by_dataset)} dataset(s) × variable model counts)."
	)
	print("Note: this stage can be slow because GNNExplainer runs per target node.")

	workers = int(config.get("extract_workers", 1) or 1)
	if workers < 1:
		workers = 1
	if workers > 1:
		requested_device = resolve_device(config.get("device"))
		print(f"Parallel extraction enabled: {workers} worker process(es).")
		print("Note: each worker loads its own copy of the graph and model (higher RAM use).")
		if requested_device == "cuda" and not enable_large_graph_cpu_fallback:
			print("Warning: parallel extraction on GPU can cause high memory use; reduce workers if needed.")

	progress_bar = None
	if tqdm is not None and total > 0:
		progress_bar = tqdm(total=total, desc="Extraction", unit="node")

	completed = 0
	records = []
	if workers == 1:
		for dataset_name, data in datasets.items():
			bundle = model_bundle.get(dataset_name)
			if bundle is None:
				continue
			device = _resolve_runtime_device_for_data(
				config.get("device"),
				data,
				enable_large_graph_cpu_fallback=enable_large_graph_cpu_fallback,
			)
			for model in bundle.values():
				model.to(device)
			if hasattr(data, "to"):
				data = data.to(device)
				datasets[dataset_name] = data
			target_nodes = target_nodes_by_dataset[dataset_name]
			for model_name, model in bundle.items():
				for node_id in target_nodes:
					if progress_bar is not None:
						progress_bar.set_postfix_str(f"{model_name}|{dataset_name}|node={node_id}")
						progress_bar.refresh()
					else:
						# Fallback progress indicator (prints ~20 times max).
						step = max(1, total // 20)
						if completed == 0 or completed % step == 0:
							pct = 100.0 * completed / total
							print(f"Extraction progress: {completed}/{total} ({pct:.1f}%)")

					bundle = extract_all(model, data, node_id, num_hops=num_hops)
					recon_cfg = config.get("reconstruction", {}) or {}
					candidate_ratio = recon_cfg.get("candidate_ratio", 4)
					max_candidates = recon_cfg.get("max_candidates")
					seed = config.get("seed")
					neighbors = get_one_hop_neighbors(data, node_id)
					bundle["candidate_set"] = build_candidate_set(
						data,
						node_id,
						neighbors,
						candidate_ratio=candidate_ratio,
						max_candidates=max_candidates,
						seed=seed,
					)
					bundle["dataset"] = dataset_name
					bundle["model"] = model_name
					records.append(
						{
							"dataset": dataset_name,
							"model": model_name,
							"target_node": int(node_id),
							"bundle": bundle,
						}
					)

					completed += 1
					if progress_bar is not None:
						progress_bar.update(1)
	else:


		output_dir = Path(str(config.get("output_dir", "outputs")))
		tmp_dir = output_dir / "_tmp_parallel_extraction"
		tmp_dir.mkdir(parents=True, exist_ok=True)

		# Save trained weights so each worker can load them without pickling the whole model.
		state_paths = {}
		model_configs = {}
		for dataset_name, bundle in model_bundle.items():
			data = datasets.get(dataset_name)
			if data is None:
				continue
			device = _resolve_runtime_device_for_data(
				config.get("device"),
				data,
				enable_large_graph_cpu_fallback=enable_large_graph_cpu_fallback,
			)
			model_configs[dataset_name] = _build_model_config_for_data(config, data)
			for model_name, model in bundle.items():
				path = tmp_dir / f"{dataset_name}__{model_name}.pt"
				torch.save(model.state_dict(), path)
				state_paths[(dataset_name, model_name)] = str(path)

		dataset_kwargs_all = config.get("dataset_kwargs", {})
		if not isinstance(dataset_kwargs_all, dict):
			dataset_kwargs_all = {}

		seed = config.get("seed")
		per_worker_threads = 1

		futures = []
		ctx = mp.get_context("spawn")
		with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
			for dataset_name in datasets.keys():
				dataset_kwargs = dataset_kwargs_all.get(dataset_name, {})
				bundle = model_bundle.get(dataset_name)
				if bundle is None:
					continue
				device = _resolve_runtime_device_for_data(
					config.get("device"),
					datasets[dataset_name],
					enable_large_graph_cpu_fallback=enable_large_graph_cpu_fallback,
				)
				model_config = model_configs[dataset_name]
				for model_name in bundle.keys():
					state_dict_path = state_paths[(dataset_name, model_name)]
					for node_id in target_nodes_by_dataset[dataset_name]:
						futures.append(
							executor.submit(
								extract_one,
								dataset_name,
								dataset_kwargs,
								model_name,
								model_config,
								state_dict_path,
								int(node_id),
								int(num_hops),
								device=device,
								torch_num_threads=per_worker_threads,
								seed=int(seed) if seed is not None else None,
							)
						)

			for future in as_completed(futures):
				record = future.result()
				bundle = record.get("bundle") or {}
				recon_cfg = config.get("reconstruction", {}) or {}
				candidate_ratio = recon_cfg.get("candidate_ratio", 4)
				max_candidates = recon_cfg.get("max_candidates")
				seed = config.get("seed")
				data = datasets.get(record.get("dataset"))
				if data is not None:
					neighbors = get_one_hop_neighbors(data, record.get("target_node"))
					bundle["candidate_set"] = build_candidate_set(
						data,
						record.get("target_node"),
						neighbors,
						candidate_ratio=candidate_ratio,
						max_candidates=max_candidates,
						seed=seed,
					)
				bundle["dataset"] = record.get("dataset")
				bundle["model"] = record.get("model")
				record["bundle"] = bundle
				records.append(record)
				completed += 1
				if progress_bar is not None:
					progress_bar.set_postfix_str(
						f"{record.get('model')}|{record.get('dataset')}|node={record.get('target_node')}"
					)
					progress_bar.update(1)
				else:
					step = max(1, total // 20)
					if completed == 1 or completed % step == 0 or completed == total:
						pct = 100.0 * completed / total
						print(f"Extraction progress: {completed}/{total} ({pct:.1f}%)")

	if progress_bar is not None:
		progress_bar.close()

	return records


def _format_subgraph_text(subgraph):
	"""Convert an extracted subgraph (often a dict) into readable text."""
	if isinstance(subgraph, dict):
		num_nodes = subgraph.get("num_nodes", "unknown")
		num_edges = subgraph.get("num_edges", "unknown")
		num_hops = subgraph.get("num_hops", "unknown")
		return f"Subgraph (k={num_hops}) with {num_nodes} nodes and {num_edges} edges."
	return str(subgraph)


def _format_raw_features_text(raw_features):
	if raw_features is None:
		return "(none)"
	return "[" + ", ".join(f"{float(v):.4f}" for v in raw_features) + "]"


def _format_neighbor_table_text(neighbor_table):
	if not neighbor_table:
		return "(none)"
	neighbor_ids = neighbor_table.get("neighbor_ids", [])
	features = neighbor_table.get("features", [])
	rows = []
	for idx, node_id in enumerate(neighbor_ids):
		if idx < len(features):
			feat_vec = features[idx]
			feat_text = "[" + ", ".join(f"{float(v):.4f}" for v in feat_vec) + "]"
		else:
			feat_text = "[]"
		rows.append(f"node {node_id}: {feat_text}")
	return "\n".join(rows) if rows else "(none)"


def _format_edge_list_text(subgraph):
	if not isinstance(subgraph, dict):
		return "(none)"
	edge_index = subgraph.get("edge_index")
	if edge_index is None:
		return "(none)"
	try:
		rows = []
		for src, dst in zip(edge_index[0], edge_index[1]):
			rows.append(f"{int(src)} -> {int(dst)}")
		return "\n".join(rows) if rows else "(none)"
	except Exception:
		return "(none)"


def _format_candidate_set_text(candidate_set):
	if not candidate_set:
		return "(none)"
	candidates = candidate_set.get("candidates", [])
	return ", ".join(str(int(v)) for v in candidates) if candidates else "(none)"


def _run_baseline_random(candidate_set, seed=None):
	import random
	rng = random.Random(seed)
	candidates = candidate_set.get("candidates", [])
	true_neighbors = candidate_set.get("true_neighbors", [])
	if not candidates:
		return []
	k = min(len(true_neighbors), len(candidates)) if true_neighbors else max(1, len(candidates) // 4)
	return rng.sample(list(candidates), k)


def _run_baseline_cosine(embedding, neighbor_table, candidate_set):
	try:
		import numpy as np
	except Exception:
		return []
	if embedding is None:
		return []
	candidates = candidate_set.get("candidates", [])
	neighbor_ids = neighbor_table.get("neighbor_ids", [])
	features = neighbor_table.get("features", [])
	if not candidates or not neighbor_ids or len(features) == 0:
		return []
	feature_map = {int(node_id): features[idx] for idx, node_id in enumerate(neighbor_ids)}
	vec = np.array(embedding, dtype=float)
	vec_norm = np.linalg.norm(vec) + 1e-8
	scores = []
	for node_id in candidates:
		feat = feature_map.get(int(node_id))
		if feat is None:
			continue
		feat = np.array(feat, dtype=float)
		score = float(np.dot(vec, feat) / (vec_norm * (np.linalg.norm(feat) + 1e-8)))
		scores.append((score, int(node_id)))
	if not scores:
		return []
	scores.sort(reverse=True)
	true_neighbors = candidate_set.get("true_neighbors", [])
	k = min(len(true_neighbors), len(scores)) if true_neighbors else max(1, len(scores) // 4)
	return [node_id for _, node_id in scores[:k]]


def _run_baseline_feature_distance(raw_features, neighbor_table, candidate_set):
	try:
		import numpy as np
	except Exception:
		return []
	if raw_features is None:
		return []
	candidates = candidate_set.get("candidates", [])
	neighbor_ids = neighbor_table.get("neighbor_ids", [])
	features = neighbor_table.get("features", [])
	if not candidates or not neighbor_ids or len(features) == 0:
		return []
	feature_map = {int(node_id): features[idx] for idx, node_id in enumerate(neighbor_ids)}
	vec = np.array(raw_features, dtype=float)
	scores = []
	for node_id in candidates:
		feat = feature_map.get(int(node_id))
		if feat is None:
			continue
		dist = float(np.linalg.norm(vec - np.array(feat, dtype=float)))
		scores.append((dist, int(node_id)))
	if not scores:
		return []
	scores.sort()
	true_neighbors = candidate_set.get("true_neighbors", [])
	k = min(len(true_neighbors), len(scores)) if true_neighbors else max(1, len(scores) // 4)
	return [node_id for _, node_id in scores[:k]]


def run_experiment_stage(config, extraction_records):
	"""Run experiment branches and return prompts/predictions by experiment."""
	experiment_names = config.get("experiments", []) or []
	if not experiment_names:
		raise ValueError("No experiments specified. Set config['experiments'] or pass a config file.")

	llm_names = config.get("llms", []) or []
	device = resolve_device(config.get("device"))
	generation_cfg = config.get("generation", {}) or {}

	outputs = {}

	for experiment in experiment_names:
		if experiment == "embedding_classification":
			prompt_cfg = config.get("prompt", {})
			template = prompt_cfg.get("template", "{explanation}\n{embedding}\n{subgraph}")
			embedding_max_length = prompt_cfg.get("embedding_max_length")
			prompts = []
			for record in extraction_records:
				bundle = record["bundle"]
				edge_mask = bundle.get("explanation_mask", {}).get("edge_mask")
				embedding = bundle.get("embedding", {}).get("embedding")
				subgraph = bundle.get("k_hop_subgraph")
				if edge_mask is None:
					explanation_text = "No explanation edge mask available."
				else:
					explanation_text = format_explanation(torch.tensor(edge_mask))
				embedding_text = format_embedding(embedding, max_length=embedding_max_length)
				subgraph_text = _format_subgraph_text(subgraph)
				prompts.append(build_classification_prompt(explanation_text, embedding_text, subgraph_text, template))
			if not llm_names:
				raise ValueError("LLM list is empty for embedding_classification experiment.")
			predictions = run_inference_all(llm_names, prompts, device, **generation_cfg)
			outputs[experiment] = {"prompts": prompts, "predictions": predictions}
			continue

		if experiment == "raw_graph_reasoning":
			prompt_cfg = config.get("prompt_raw_reasoning", {})
			template = prompt_cfg.get("template", "{raw_features}\n{neighbor_table}\n{edge_list}")
			prompts = []
			for record in extraction_records:
				bundle = record["bundle"]
				raw_features_text = _format_raw_features_text(bundle.get("raw_features"))
				neighbor_table_text = _format_neighbor_table_text(bundle.get("neighbor_feature_table"))
				edge_list_text = _format_edge_list_text(bundle.get("k_hop_subgraph"))
				prompts.append(build_raw_reasoning_prompt(raw_features_text, neighbor_table_text, edge_list_text, template))
			if not llm_names:
				raise ValueError("LLM list is empty for raw_graph_reasoning experiment.")
			predictions = run_inference_all(llm_names, prompts, device, **generation_cfg)
			outputs[experiment] = {"prompts": prompts, "predictions": predictions}
			continue

		if experiment == "reconstruction_1hop":
			prompt_cfg = config.get("prompt_reconstruction", {})
			template = prompt_cfg.get("template", "{embedding}\n{candidates}")
			prompts = []
			for record in extraction_records:
				bundle = record["bundle"]
				embedding = bundle.get("embedding", {}).get("embedding")
				embedding_text = format_embedding(embedding, max_length=None)
				candidate_text = _format_candidate_set_text(bundle.get("candidate_set"))
				prompts.append(build_neighbor_selection_prompt(embedding_text, candidate_text, template))
			if not llm_names:
				raise ValueError("LLM list is empty for reconstruction_1hop experiment.")
			predictions = run_inference_all(llm_names, prompts, device, **generation_cfg)
			outputs[experiment] = {"prompts": prompts, "predictions": predictions}
			continue

		if experiment == "baseline_random":
			rows = []
			seed = config.get("seed")
			for record in extraction_records:
				bundle = record["bundle"]
				candidate_set = bundle.get("candidate_set") or {}
				predicted = _run_baseline_random(candidate_set, seed=seed)
				rows.append(
					{
						"dataset": record["dataset"],
						"model": record["model"],
						"target_node": record["target_node"],
						"true_neighbors": candidate_set.get("true_neighbors", []),
						"predicted_neighbors": predicted,
					}
				)
			outputs[experiment] = {"baseline": rows}
			continue

		if experiment == "baseline_cosine":
			rows = []
			for record in extraction_records:
				bundle = record["bundle"]
				candidate_set = bundle.get("candidate_set") or {}
				predicted = _run_baseline_cosine(
					bundle.get("embedding", {}).get("embedding"),
					bundle.get("neighbor_feature_table", {}),
					candidate_set,
				)
				rows.append(
					{
						"dataset": record["dataset"],
						"model": record["model"],
						"target_node": record["target_node"],
						"true_neighbors": candidate_set.get("true_neighbors", []),
						"predicted_neighbors": predicted,
					}
				)
			outputs[experiment] = {"baseline": rows}
			continue

		if experiment == "baseline_feature":
			rows = []
			for record in extraction_records:
				bundle = record["bundle"]
				candidate_set = bundle.get("candidate_set") or {}
				predicted = _run_baseline_feature_distance(
					bundle.get("raw_features"),
					bundle.get("neighbor_feature_table", {}),
					candidate_set,
				)
				rows.append(
					{
						"dataset": record["dataset"],
						"model": record["model"],
						"target_node": record["target_node"],
						"true_neighbors": candidate_set.get("true_neighbors", []),
						"predicted_neighbors": predicted,
					}
				)
			outputs[experiment] = {"baseline": rows}
			continue

		raise ValueError(f"Unknown experiment: {experiment}")

	return outputs


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
			prediction_bundle = (record.get("bundle") or {}).get("prediction") or {}
			gnn_pred = prediction_bundle.get("predicted_class")
			if gnn_pred is None:
				raise KeyError(
					"Missing predicted_class in extraction record bundle: "
					"expected record['bundle']['prediction']['predicted_class']."
				)
			llm_pred = llm_preds[idx]
			comparisons.append(
				{
					"dataset": record["dataset"],
					"model": record["model"],
					"llm": llm_name,
					"target_node": record["target_node"],
					"target_class": prediction_bundle.get("target_class"),
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


def run_evaluation_stage_experiments(config, extraction_records, experiment_outputs):
	"""Evaluate each experiment branch and save results."""
	output_dir = Path(str(config.get("output_dir", "outputs")))
	output_dir.mkdir(parents=True, exist_ok=True)

	results = {}
	for experiment, payload in experiment_outputs.items():
		if "predictions" in payload:
			prompts = payload.get("prompts", [])
			predictions_by_llm = payload.get("predictions", {})
			comparisons = []
			for llm_name, llm_preds in predictions_by_llm.items():
				for idx, record in enumerate(extraction_records):
					prediction_bundle = (record.get("bundle") or {}).get("prediction") or {}
					gnn_pred = prediction_bundle.get("predicted_class")
					llm_pred = llm_preds[idx] if idx < len(llm_preds) else None
					comparisons.append(
						{
							"experiment": experiment,
							"dataset": record["dataset"],
							"model": record["model"],
							"llm": llm_name,
							"target_node": record["target_node"],
							"target_class": prediction_bundle.get("target_class"),
							"gnn_pred": gnn_pred,
							"llm_pred": llm_pred,
							"prompt": prompts[idx] if idx < len(prompts) else None,
						}
					)

			if experiment == "reconstruction_1hop":
				rows = []
				for llm_name, llm_preds in predictions_by_llm.items():
					for idx, record in enumerate(extraction_records):
						bundle = record.get("bundle") or {}
						candidate_set = bundle.get("candidate_set") or {}
						pred = llm_preds[idx] if idx < len(llm_preds) else None
						rows.append(
							{
								"dataset": record["dataset"],
								"model": record["model"],
								"llm": llm_name,
								"target_node": record["target_node"],
								"true_neighbors": candidate_set.get("true_neighbors", []),
								"predicted_neighbors": parse_neighbor_selection_response(str(pred)),
							}
						)

				metrics = evaluate_reconstruction(rows)
				summary_path = str(output_dir / f"results_summary_{experiment}.json")
				raw_path = str(output_dir / f"results_raw_{experiment}.json")
				save_results(metrics, summary_path, fmt="json")
				save_results(rows, raw_path, fmt="json")
				results[experiment] = {
					"summary": metrics,
					"comparisons": rows,
					"paths": {"summary": summary_path, "raw": raw_path},
				}
				continue

			grouped = {}
			for row in comparisons:
				group = f"{row['experiment']}|{row['model']}|{row['dataset']}|{row['llm']}"
				grouped.setdefault(group, []).append({"gnn_pred": row["gnn_pred"], "llm_pred": row["llm_pred"]})

			summary = {key: compute_classification_metrics(rows) for key, rows in grouped.items()}
			raw_path = str(output_dir / f"results_raw_{experiment}.json")
			summary_path = str(output_dir / f"results_summary_{experiment}.json")
			save_results(comparisons, raw_path, fmt="json")
			save_results(summary, summary_path, fmt="json")
			results[experiment] = {
				"summary": summary,
				"comparisons": comparisons,
				"paths": {"summary": summary_path, "raw": raw_path},
			}
			continue

		if "baseline" in payload:
			rows = payload.get("baseline", [])
			metrics = evaluate_reconstruction(rows)
			summary_path = str(output_dir / f"results_summary_{experiment}.json")
			raw_path = str(output_dir / f"results_raw_{experiment}.json")
			save_results(metrics, summary_path, fmt="json")
			save_results(rows, raw_path, fmt="json")
			results[experiment] = {
				"summary": metrics,
				"comparisons": rows,
				"paths": {"summary": summary_path, "raw": raw_path},
			}
			continue

		raise ValueError(f"Unknown experiment payload format for {experiment}")

	return results


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
		from Final_version.Train import set_seed
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
		if merged.get("experiments"):
			llm_outputs = run_experiment_stage(merged, extraction_records)
			state["llm"] = llm_outputs
		else:
			llm_outputs = run_llm_stage(merged, extraction_records)
			state["llm"] = llm_outputs
	else:
		state["llm"] = None

	if stages["eval"]:
		if extraction_records is None:
			raise RuntimeError("Evaluation requested but extraction records are missing.")
		if llm_outputs is None:
			raise RuntimeError("Evaluation requested but LLM outputs are missing.")
		if merged.get("experiments"):
			evaluation = run_evaluation_stage_experiments(merged, extraction_records, llm_outputs)
			state["evaluation"] = evaluation
		else:
			evaluation = run_evaluation_stage(merged, extraction_records, llm_outputs)
			state["evaluation"] = evaluation
	else:
		state["evaluation"] = None

	return state


def main(argv=None):
	"""CLI entry point."""
	args = parse_args(argv)
	config = load_config(args.config)

	num_runs = int(getattr(args, "num_runs", 1) or 1)
	if num_runs < 1:
		raise ValueError("--num-runs must be >= 1")

	for run_index in range(num_runs):
		run_id = _resolve_run_id(args, run_index)
		per_run_args = copy.deepcopy(args)
		per_run_args.seed = _resolve_run_seed(config, args, run_index)
		per_run_args.output_dir = _resolve_run_output_dir(config, args, run_id)

		if num_runs > 1 or run_id is not None:
			label = f"Run {run_index + 1}/{num_runs}"
			if run_id is not None:
				label += f" (id={run_id})"
			print(f"=== {label} ===")
			if per_run_args.seed is not None:
				print(f"Seed: {per_run_args.seed}")
			if per_run_args.output_dir is not None:
				print(f"Output dir: {per_run_args.output_dir}")

		state = run_pipeline(config, per_run_args)

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
