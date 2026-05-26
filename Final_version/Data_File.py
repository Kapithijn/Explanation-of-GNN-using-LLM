from pathlib import Path

from torch_geometric.data import Data
from torch_geometric.datasets import EllipticBitcoinDataset, EllipticBitcoinTemporalDataset, DGraphFin
import torch

DATA_ROOT = Path(__file__).parent / "data"
ELLIPTIC_ROOT = DATA_ROOT / "elliptic_dataset"
ELLIPTIC_TEMPORAL_ROOT = DATA_ROOT / "elliptic_temporal_dataset"
DGRAPHFIN_ROOT = DATA_ROOT / "dgraphfin_dataset"


def _to_data(dataset):
	"""Return the first graph stored in a PyG dataset object."""
	if len(dataset) == 0:
		raise ValueError("The dataset did not contain any graph data.")
	return dataset[0]


def load_elliptic(root=ELLIPTIC_ROOT, force_reload=False):
	"""Load the static Elliptic Bitcoin dataset."""
	dataset = EllipticBitcoinDataset(root=str(root), force_reload=force_reload)
	return _to_data(dataset)


def load_elliptic_temporal(t=1, root=ELLIPTIC_TEMPORAL_ROOT, force_reload=False):
	"""Load the time-step aware Elliptic Bitcoin dataset.

	The PyG dataset requires an explicit timestep between 1 and 49.
	"""
	if t < 1 or t > 49:
		raise ValueError("EllipticBitcoinTemporalDataset expects t to be between 1 and 49.")

	dataset = EllipticBitcoinTemporalDataset(root=str(root), t=t, force_reload=force_reload)
	return _to_data(dataset)


def load_dgraphfin(root=DGRAPHFIN_ROOT, force_reload=False):
	"""Load the DGraphFin dynamic financial graph dataset."""
	raw_zip = Path(root) / "raw" / "DGraphFin.zip"
	if not raw_zip.exists():
		raise FileNotFoundError(
			f"Missing DGraphFin dataset archive: {raw_zip}. "
			"Download 'DGraphFin.zip' from https://dgraph.xinye.com and place it in the raw directory."
		)
	dataset = DGraphFin(root=str(root), force_reload=force_reload)
	return _to_data(dataset)


def load_dataset(name, **kwargs):
	"""Load one of the supported graph datasets by name."""
	normalized_name = name.strip().lower()

	if normalized_name in {"elliptic", "ellipticbitcoin", "ellipticbitcoindataset"}:
		return load_elliptic(**kwargs)
	if normalized_name in {"elliptictemp", "elliptic_temporal", "ellipticbitcointemporaldataset"}:
		return load_elliptic_temporal(**kwargs)
	if normalized_name in {"dgraph", "dgraphfin", "dgraphfindataset"}:
		return load_dgraphfin(**kwargs)

	raise ValueError(f"Unknown dataset name: {name}")


def preprocess(data):
	"""Validate a graph data object and return it unchanged.

	These datasets are already provided as graph objects, so loading is the
	main dataset-specific step. This hook keeps the pipeline interface stable.
	"""
	if not isinstance(data, Data):
		raise TypeError("preprocess expects a torch_geometric.data.Data object")

	if not hasattr(data, "x") or not hasattr(data, "edge_index"):
		raise ValueError("The data object must contain node features and an edge_index.")

	return data


def print_data_info(data):
	"""Print basic statistics for a graph dataset."""
	if not isinstance(data, Data):
		raise TypeError("print_data_info expects a torch_geometric.data.Data object")

	num_nodes = data.num_nodes
	num_edges = data.edge_index.size(1) if data.edge_index is not None else 0
	num_features = data.num_node_features

	print(f"Nodes: {num_nodes}, Edges: {num_edges}, Features: {num_features}")

	if getattr(data, "y", None) is not None:
		labels = data.y
		valid_labels = labels[labels >= 0] if labels.numel() > 0 else labels
		if valid_labels.numel() > 0:
			unique_labels, counts = torch.unique(valid_labels, return_counts=True)
			class_info = {int(label.item()): int(count.item()) for label, count in zip(unique_labels, counts)}
			print(f"Label counts: {class_info}")

	for mask_name in ("train_mask", "val_mask", "test_mask"):
		mask = getattr(data, mask_name, None)
		if mask is not None:
			print(f"{mask_name}: {int(mask.sum().item())}")

	print(f"Any NaN in X: {torch.isnan(data.x).any().item()}")
	print(f"Any Inf in X: {torch.isinf(data.x).any().item()}")

