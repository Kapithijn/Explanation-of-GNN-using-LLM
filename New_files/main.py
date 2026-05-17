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
from Data_File import load_dataset, preprocess, print_data_info
from GNN_Definition import build_model_bundle
from Train import train_all
from LLM_Module import format_explanation, format_embedding, build_prompt, run_inference_all
from Evalueation import aggregate_results, save_results
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from Parallel_Extraction import extract_one

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
		"datasets": ["Dgraph"],#elliptic
		"models": ["GIN", "GraphSAGE"],  # subset of bundle keys or None for all
		"llms": ["Qwen/Qwen2.5-3B-Instruct"],#Qwen/Qwen3.5-2B or Qwen/Qwen2.5-0.5B-Instruct Qwen2.5-3B-Instruct
		"extract_workers": 2,
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
				"What class does this node belong to? return 1 if positive, 0 if negative."
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
	from Extracion import extract_all

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
