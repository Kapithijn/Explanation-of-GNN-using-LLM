import torch
from torch_geometric.explain import GNNExplainer
from torch_geometric.explain.config import ExplainerConfig, ModelConfig

def get_node_embeddings(model, data, layer="hidden"):
	"""Return node representations for all nodes.

	layer='hidden' returns the representation after the first message-passing
	layer. layer='logits' returns the final output of the model.
	"""
	if layer not in {"hidden", "logits"}:
		raise ValueError("layer must be one of {'hidden', 'logits'}")

	model.eval()
	with torch.no_grad():
		if layer == "hidden":
			_, embeddings = model(data, return_hidden=True)
		else:
			embeddings = model(data)

	return embeddings


def get_target_node_embedding(model, data, target_node_idx, layer="hidden"):
	"""Return one target-node embedding as a 1D tensor."""
	embeddings = get_node_embeddings(model, data, layer=layer)
	return embeddings[target_node_idx]


def get_target_node_embedding_all_models(model_bundle, data, target_node_idx, layer="hidden"):
	"""Return target-node embeddings for every model in model_bundle."""
	result = {}
	for model_name, parts in model_bundle.items():
		embedding = get_target_node_embedding(
			parts["model"],
			data,
			target_node_idx=target_node_idx,
			layer=layer,
		)
		result[model_name] = embedding.detach().cpu()
	return result



def get_explanation(model, data, target_node_idx, layer="hidden"):
	"""Return an explanation for the target node."""

	model.eval()
	explainer = GNNExplainer(epochs=200)
	explainer.connect(
		ExplainerConfig(
			explanation_type="phenomenon",
			node_mask_type="attributes",
			edge_mask_type="object",
		),
		ModelConfig(
				mode="multiclass_classification",
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
            
	subgraph_data = explanation.get_explanation_subgraph()
	return node_feat_mask, explanation.edge_mask, subgraph_data

def get_explanation_all_models(model_bundle, data, target_node_idx, layer="hidden"):
    """Return explanations for the target node for every model in model_bundle."""
    result = {}
    for model_name, parts in model_bundle.items():
        node_feat_mask, edge_mask, subgraph_data = get_explanation(
            parts["model"],
            data,
            target_node_idx=target_node_idx,
            layer=layer,
        )
        result[model_name] = {
            "node_feat_mask": node_feat_mask.detach().cpu() if node_feat_mask is not None else None,
            "edge_mask": edge_mask.detach().cpu() if edge_mask is not None else None,
            "subgraph_data": subgraph_data, 
        }
    return result