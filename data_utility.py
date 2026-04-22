import os
from torch_geometric.data import Data
import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def load_data(path="transaction_dataset.csv"):
    """Loads the transaction dataset from a CSV file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected dataset file '{path}' not found."
                            " Please ensure the dataset is in the current directory."
                            " You can download it from the provided link."
                            " If you have already downloaded it, please check the filename and path.")
    return pd.read_csv(path)

def preprocess_data(data, imputer=None, scaler=None, test_size=0.2, random_state=42):
    target_col = "FLAG"
    if target_col not in data.columns:
        raise ValueError("Expected target column 'FLAG' in dataset")

    leakage_cols = ["Unnamed: 0", "Index"]
    feature_df = data.drop(columns=[target_col] + leakage_cols, errors="ignore")

    num_df = feature_df.select_dtypes(include="number")

    # Handle case with no numeric features by creating a column of zeros.
    if num_df.shape[1] == 0:
        X = torch.zeros((len(data), 1), dtype=torch.float)
        y_array = data[target_col].to_numpy()
        all_idx = np.arange(len(data))
        train_idx_np, test_idx_np = train_test_split(
            all_idx,
            test_size=test_size,
            random_state=random_state,
            stratify=y_array,
        )
        train_idx = torch.tensor(train_idx_np, dtype=torch.long)
        test_idx = torch.tensor(test_idx_np, dtype=torch.long)
        if imputer is None:
            imputer = SimpleImputer(strategy="median")
        if scaler is None:
            scaler = StandardScaler()
    else:
        y_array = data[target_col].to_numpy()
        all_idx = np.arange(len(data))

        train_idx_np, test_idx_np = train_test_split(
            all_idx,
            test_size=test_size,
            random_state=random_state,
            stratify=y_array,
        )

        if imputer is None:
            imputer = SimpleImputer(strategy="median")
            train_features = imputer.fit_transform(num_df.iloc[train_idx_np])
        else:
            train_features = imputer.transform(num_df.iloc[train_idx_np])

        if scaler is None:
            scaler = StandardScaler()
            train_features = scaler.fit_transform(train_features)
        else:
            train_features = scaler.transform(train_features)

        all_features = scaler.transform(imputer.transform(num_df))
        X = torch.tensor(all_features, dtype=torch.float)

        train_idx = torch.tensor(train_idx_np, dtype=torch.long)
        test_idx = torch.tensor(test_idx_np, dtype=torch.long)

    # Graph
    num_nodes = X.size(0)
    if num_nodes >= 2:
        src = torch.arange(0, num_nodes - 1, dtype=torch.long)
        dst = torch.arange(1, num_nodes, dtype=torch.long)
        edge_index = torch.stack([src, dst], dim=0)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    y = torch.tensor(data[target_col].values, dtype=torch.long)

    graph_data = Data(x=X, edge_index=edge_index, y=y)

    graph_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    graph_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    graph_data.train_mask[train_idx] = True
    graph_data.test_mask[test_idx] = True

    return graph_data, train_idx, test_idx, imputer, scaler

def load_and_preprocess_data(path="transaction_dataset.csv", imputer=None, scaler=None):
    data = load_data(path)
    return preprocess_data(data, imputer=imputer, scaler=scaler)

def print_data_stats(graph_data):
    """Prints basic statistics about the graph data."""
    num_nodes = graph_data.num_nodes
    num_features = graph_data.num_node_features
    num_classes = int(graph_data.y.max().item()) + 1
    train_pos = int(graph_data.y[graph_data.train_mask].sum().item())
    train_total = int(graph_data.train_mask.sum().item())
    test_pos = int(graph_data.y[graph_data.test_mask].sum().item())
    test_total = int(graph_data.test_mask.sum().item())
    majority_baseline = max(train_total - train_pos, train_pos) / train_total if train_total > 0 else 0
    print(f"Nodes: {num_nodes}, Features: {num_features}, Classes: {num_classes}")
    print(f"Train class-1 rate: {train_pos / train_total:.4f}, Test class-1 rate: {test_pos / test_total:.4f}")
    print(f"Majority-class baseline accuracy on train split: {majority_baseline:.4f}")
    print(f"Any NaN in X: {torch.isnan(graph_data.x).any().item()}, Any Inf in X: {torch.isinf(graph_data.x).any().item()}")
    print(f"duplicate nodes in test and train: {(graph_data.train_mask & graph_data.test_mask).any().item()}")






# data = pd.read_csv("transaction_dataset.csv")

# target_col = "FLAG"
# if target_col not in data.columns:
#     raise ValueError("Expected target column 'FLAG' in transaction_dataset.csv")

# # Drop obvious leakage / identifier columns if present.
# leakage_cols = ["Unnamed: 0", "Index"]
# feature_df = data.drop(columns=[target_col] + leakage_cols, errors="ignore")

# # Keep only numeric features for the GCN input.
# num_df = feature_df.select_dtypes(include="number")
# if num_df.shape[1] == 0:
#     X = torch.zeros((len(data), 1), dtype=torch.float)
# else:
#     y_array = data[target_col].to_numpy()
#     all_idx = np.arange(len(data))
#     train_idx_np, test_idx_np = train_test_split(
#         all_idx,
#         test_size=0.2,
#         random_state=42,
#         stratify=y_array,
#     )

#     # Fit preprocessing on train rows only to avoid test leakage.
#     imputer = SimpleImputer(strategy="median")
#     scaler = StandardScaler()

#     train_features = imputer.fit_transform(num_df.iloc[train_idx_np])
#     train_features = scaler.fit_transform(train_features)
#     all_features = scaler.transform(imputer.transform(num_df))

#     X = torch.tensor(all_features, dtype=torch.float)

#     train_idx = torch.tensor(train_idx_np, dtype=torch.long)
#     test_idx = torch.tensor(test_idx_np, dtype=torch.long)

# # Build a simple valid edge_index (chain graph: 0->1->2->...)
# num_nodes = X.size(0)
# if num_nodes >= 2:
#     src = torch.arange(0, num_nodes - 1, dtype=torch.long)
#     dst = torch.arange(1, num_nodes, dtype=torch.long)
#     edge_index = torch.stack([src, dst], dim=0)
# else:
#     edge_index = torch.empty((2, 0), dtype=torch.long)

# # Add labels
# y = torch.tensor(data[target_col].values, dtype=torch.long)

# # Build graph data object
# train_data = Data(x=X, edge_index=edge_index, y=y)

# # Train/test split masks
# num_nodes = train_data.num_nodes
# train_data.train_mask = torch.zeros(num_nodes, dtype=torch.bool)
# train_data.test_mask = torch.zeros(num_nodes, dtype=torch.bool)

# train_data.train_mask[train_idx] = True
# train_data.test_mask[test_idx] = True

# train_pos = int(train_data.y[train_data.train_mask].sum().item())
# train_total = int(train_data.train_mask.sum().item())
# test_pos = int(train_data.y[train_data.test_mask].sum().item())
# test_total = int(train_data.test_mask.sum().item())
# majority_baseline = max(train_total - train_pos, train_pos) / train_total if train_total > 0 else 0

# print(f"Nodes: {num_nodes}, Features: {X.size(1)}, Classes: {int(y.max().item()) + 1}")
# print(f"Train class-1 rate: {train_pos / train_total:.4f}, Test class-1 rate: {test_pos / test_total:.4f}")
# print(f"Majority-class baseline accuracy on train split: {majority_baseline:.4f}")
# print(f"Any NaN in X: {torch.isnan(X).any().item()}, Any Inf in X: {torch.isinf(X).any().item()}")
# print(f"duplicate nodes in test and train: {(train_data.train_mask & train_data.test_mask).any().item()}")


