import torch


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
		emb = get_target_node_embedding(
			parts["model"],
			data,
			target_node_idx=target_node_idx,
			layer=layer,
		)
		result[model_name] = emb.detach().cpu()
	return result
