import torch
from torch_geometric.explain import GNNExplainer
from torch_geometric.explain.config import ExplainerConfig, ModelConfig


def get_explanation(model, data, target_node_idx, layer="hidden"):
	"""Return an explanation for the target node."""

	explainer = GNNExplainer(epochs=200)
	explainer.connect(
		ExplainerConfig(
			explanation_type="node",
			node_mask_type="attributes",
			edge_mask_type="object",
		),
		ModelConfig(
			mode="binary_classification",
			task_level="node",
			return_type="raw",
		),
	)
	explanation = explainer(
		model,
		data.x,
		data.edge_index,
		target=data.y,
		index=target_node_idx,
	)
	node_feat_mask = explanation.node_mask
	if node_feat_mask is not None and node_feat_mask.dim() == 2:
		node_feat_mask = node_feat_mask[target_node_idx]
	return node_feat_mask, explanation.edge_mask

def get_explanation_all_models(model_bundle, data, target_node_idx, layer="hidden"):
    """Return explanations for the target node for every model in model_bundle."""
    result = {}
    for model_name, parts in model_bundle.items():
        node_feat_mask, edge_mask = get_explanation(
            parts["model"],
            data,
            target_node_idx=target_node_idx,
            layer=layer,
        )
        result[model_name] = {
            "node_feat_mask": node_feat_mask.detach().cpu() if node_feat_mask is not None else None,
            "edge_mask": edge_mask.detach().cpu() if edge_mask is not None else None,
        }
    return result