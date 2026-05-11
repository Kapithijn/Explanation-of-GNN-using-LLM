import torch
import torch.nn.functional as F
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.utils import k_hop_subgraph


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
    
    return {
        "node_id": target_node,
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
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
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
    
    return {
        "target_node": target_node,
        "prediction": prediction,
        "explanation": explanation,
        "embedding": embedding,
        "subgraph": subgraph,
    }
