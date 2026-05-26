import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph
import numpy as np


def get_prediction(model, data, target_node):
    """
    Returns the GNN's class prediction for the target node.
    
    Args:
        model: Trained GNN model
        data: PyG Data object
        target_node: Index of the target node
    
    Returns:
        Dictionary with 'node_id', 'logits', and 'predicted_class'
    """
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        logits = out[target_node]
        predicted_class = logits.argmax(dim=0).item()

    target_class = None
    y = getattr(data, "y", None)
    if y is not None:
        try:
            target_class = int(y[target_node].item())
        except Exception:
            target_class = None
    
    return {
        "node_id": target_node,
        "target_class": target_class,
        "logits": logits.cpu().numpy(),
        "predicted_class": predicted_class,
    }


def get_explanation(model, data, target_node):
    """
    Runs GNNExplainer and returns edge/feature importance masks.
    
    Args:
        model: Trained GNN model
        data: PyG Data object
        target_node: Index of the target node
    
    Returns:
        Dictionary with 'edge_mask' and 'feature_mask' (numpy arrays)
    """
    model.eval()
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=50),
        explanation_type="model",
        node_mask_type="attributes",
        edge_mask_type="object",
        model_config=dict(
            mode="multiclass_classification",
            task_level="node",
            return_type="log_probs",
        ),
    )

    explanation = explainer(data.x, data.edge_index, index=target_node)
    
    return {
        "node_id": target_node,
        "edge_mask": explanation.edge_mask.cpu().numpy() if explanation.edge_mask is not None else None,
        "feature_mask": explanation.node_mask.cpu().numpy() if explanation.node_mask is not None else None,
    }


def get_embedding(model, data, target_node):
    """
    Extracts the latent embedding of the target node.
    
    Args:
        model: Trained GNN model
        data: PyG Data object
        target_node: Index of the target node
    
    Returns:
        Dictionary with 'node_id' and 'embedding' (numpy array)
    """
    model.eval()
    
    # For GNN models with multiple layers, we extract from the second-to-last layer
    # This requires access to intermediate activations
    embedding = None
    
    with torch.no_grad():
        # Forward pass to get intermediate embeddings
        x = data.x
        edge_index = data.edge_index
        
        # For GCN, GAT, GIN, GraphSAGE, we extract after first conv layer
        # This is a general approach; adjust layer if needed
        if hasattr(model, 'conv1'):
            x = model.conv1(x, edge_index)
            x = F.relu(x)
            embedding = x[target_node]
        else:
            # Fallback: use final output if intermediate not available
            out = model(data.x, edge_index)
            embedding = out[target_node]
    
    return {
        "node_id": target_node,
        "embedding": embedding.cpu().numpy(),
        "embedding_dim": len(embedding),
    }


def get_subgraph(data, target_node, num_hops=2):
    """
    Extracts the k-hop subgraph around the target node.
    
    Args:
        data: PyG Data object
        target_node: Index of the target node
        num_hops: Number of hops to extract (default: 2)
    
    Returns:
        Dictionary with subgraph structure, node indices, and edge indices
    """
    
    subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
        node_idx=target_node,
        num_hops=num_hops,
        edge_index=data.edge_index,
        relabel_nodes=True,
    )
    
    # Extract subgraph node features
    sub_x = data.x[subset] if hasattr(data, 'x') else None
    
    # Extract subgraph labels if available
    sub_y = data.y[subset] if hasattr(data, 'y') else None
    
    return {
        "node_id": target_node,
        "num_hops": num_hops,
        "subset_nodes": subset.cpu().numpy(),
        "edge_index": sub_edge_index.cpu().numpy(),
        "node_features": sub_x.cpu().numpy() if sub_x is not None else None,
        "node_labels": sub_y.cpu().numpy() if sub_y is not None else None,
        "target_mapping": mapping.item() if mapping.numel() == 1 else mapping.cpu().numpy(),
        "num_nodes": len(subset),
        "num_edges": sub_edge_index.shape[1],
    }


def get_raw_features(data, target_node):
    """Return the raw feature vector for the target node."""
    if not hasattr(data, "x") or data.x is None:
        return None
    return data.x[target_node].detach().cpu().numpy()


def get_one_hop_neighbors(data, target_node):
    """Return a sorted list of one-hop neighbor node ids (undirected view)."""
    edge_index = getattr(data, "edge_index", None)
    if edge_index is None:
        return []
    src, dst = edge_index
    mask_src = src == int(target_node)
    mask_dst = dst == int(target_node)
    neighbors = torch.cat([dst[mask_src], src[mask_dst]]).unique()
    return sorted([int(v) for v in neighbors.detach().cpu().tolist()])


def get_neighbor_feature_table(data, neighbor_ids):
    """Return neighbor features aligned with neighbor ids for prompts."""
    if not hasattr(data, "x") or data.x is None or not neighbor_ids:
        return {"neighbor_ids": [], "features": []}
    feats = data.x[torch.tensor(neighbor_ids, device=data.x.device)]
    return {
        "neighbor_ids": [int(v) for v in neighbor_ids],
        "features": feats.detach().cpu().numpy(),
    }


def build_candidate_set(data, target_node, true_neighbors, candidate_ratio=4, max_candidates=None, seed=None):
    """Build a candidate set with true neighbors plus sampled non-neighbors."""
    num_nodes = int(getattr(data, "num_nodes", 0) or 0)
    if num_nodes <= 0:
        return {
            "true_neighbors": [int(v) for v in true_neighbors],
            "candidates": [int(v) for v in true_neighbors],
        }

    rng = np.random.default_rng(seed)
    true_set = set(int(v) for v in true_neighbors)
    true_set.discard(int(target_node))
    all_nodes = set(range(num_nodes))
    negatives = list(all_nodes.difference(true_set).difference({int(target_node)}))

    neg_count = int(len(true_set) * candidate_ratio)
    if max_candidates is not None:
        max_candidates = int(max_candidates)
        neg_count = min(neg_count, max(0, max_candidates - len(true_set)))

    if neg_count > 0 and negatives:
        if neg_count >= len(negatives):
            sampled = negatives
        else:
            sampled = rng.choice(negatives, size=neg_count, replace=False).tolist()
    else:
        sampled = []

    candidates = sorted(list(true_set) + [int(v) for v in sampled])
    return {
        "true_neighbors": sorted(list(true_set)),
        "candidates": candidates,
    }


def extract_all(model, data, target_node, num_hops=2):
    """
    Runs all extractions and returns a structured bundle containing:
    - GNN prediction
    - GNNExplainer explanation masks
    - Node embedding
    - k-hop subgraph
    
    Args:
        model: Trained GNN model
        data: PyG Data object
        target_node: Index of the target node
        num_hops: Number of hops for subgraph extraction (default: 2)
    
    Returns:
        Dictionary with all extracted components
    """
    prediction = get_prediction(model, data, target_node)
    explanation = get_explanation(model, data, target_node)
    embedding = get_embedding(model, data, target_node)
    subgraph = get_subgraph(data, target_node, num_hops=num_hops)
    raw_features = get_raw_features(data, target_node)
    one_hop_neighbors = get_one_hop_neighbors(data, target_node)
    neighbor_feature_table = get_neighbor_feature_table(data, one_hop_neighbors)
    candidate_set = build_candidate_set(data, target_node, one_hop_neighbors)
    
    return {
        "dataset": None,
        "model": None,
        "target_node": target_node,
        "ground_truth_label": prediction.get("target_class"),
        "prediction": prediction,
        "logits": prediction.get("logits"),
        "embedding": embedding,
        "embedding_dimension": embedding.get("embedding_dim"),
        "explanation": explanation,
        "explanation_mask": {
            "edge_mask": explanation.get("edge_mask"),
            "feature_mask": explanation.get("feature_mask"),
        },
        "subgraph": subgraph,
        "k_hop_subgraph": subgraph,
        "one_hop_neighbors": one_hop_neighbors,
        "raw_features": raw_features,
        "neighbor_feature_table": neighbor_feature_table,
        "candidate_set": candidate_set,
        "metadata": {},
    }
