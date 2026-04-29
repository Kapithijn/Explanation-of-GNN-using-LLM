import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GINConv
from torch_geometric.nn import SAGEConv

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index=None, return_hidden=False):
        if edge_index is None:
            data = x
            x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        hidden = x
        x = self.conv2(x, edge_index)
        if return_hidden:
            return x, hidden
        return x


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=4, dropout=0.5):
        super().__init__()
        self.conv1 = GATConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
        )
        self.conv2 = GATConv(
            hidden_channels * heads,
            out_channels,
            heads=1,
            concat=False,
            dropout=dropout,
        )
        self.dropout = dropout

    def forward(self, x, edge_index=None, return_hidden=False):
        if edge_index is None:
            data = x
            x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        hidden = x
        x = self.conv2(x, edge_index)
        if return_hidden:
            return x, hidden
        return x


class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, hidden_channels),
            )
        )
        self.conv2 = GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(hidden_channels, hidden_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_channels, out_channels),
            )
        )
        self.dropout = dropout

    def forward(self, x, edge_index=None, return_hidden=False):
        if edge_index is None:
            data = x
            x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        hidden = x
        x = self.conv2(x, edge_index)
        if return_hidden:
            return x, hidden
        return x

class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index=None, return_hidden=False):
        if edge_index is None:
            data = x
            x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        hidden = x
        x = self.conv2(x, edge_index)
        if return_hidden:
            return x, hidden
        return x



def build_model_bundle(
    in_channels,
    out_channels,
    hidden_channels=16,
    heads=4,
    dropout=0.5,
    lr=0.005,
    weight_decay=5e-4,
    device=None, models_to_include=None
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_gcn = GCN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=dropout,
    ).to(device)
    model_gat = GAT(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        heads=heads,
        dropout=dropout,
    ).to(device)
    model_gin = GIN(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,
        dropout=dropout,
    ).to(device)
    model_sage = GraphSAGE(
        in_channels=in_channels,
        hidden_channels=hidden_channels,
        out_channels=out_channels,        dropout=dropout,
    ).to(device)

    if models_to_include is not None:
        if "GCN" not in models_to_include:
            model_gcn = None
        if "GAT" not in models_to_include:
            model_gat = None
        if "GIN" not in models_to_include:
            model_gin = None
        if "GraphSAGE" not in models_to_include:
            model_sage = None
    bundle = {}
    if model_gcn is not None:
        bundle["GCN"] = {
            "model": model_gcn,
            "optimizer": torch.optim.Adam(model_gcn.parameters(), lr=lr, weight_decay=weight_decay),
            "criterion": torch.nn.CrossEntropyLoss(),
        }
    if model_gat is not None:
        bundle["GAT"] = {
            "model": model_gat,
            "optimizer": torch.optim.Adam(model_gat.parameters(), lr=lr, weight_decay=weight_decay),
            "criterion": torch.nn.CrossEntropyLoss(),
        }
    if model_gin is not None:
        bundle["GIN"] = {
            "model": model_gin,
            "optimizer": torch.optim.Adam(model_gin.parameters(), lr=lr, weight_decay=weight_decay),
            "criterion": torch.nn.CrossEntropyLoss(),
        }
    if model_sage is not None:
        bundle["GraphSAGE"] = {
            "model": model_sage,
            "optimizer": torch.optim.Adam(model_sage.parameters(), lr=lr, weight_decay=weight_decay),
            "criterion": torch.nn.CrossEntropyLoss(),
        }
    return bundle
    # bundle = {
    #     "GCN": {
    #         "model": model_gcn,
    #         "optimizer": torch.optim.Adam(model_gcn.parameters(), lr=lr, weight_decay=weight_decay),
    #         "criterion": torch.nn.CrossEntropyLoss(),
    #     },
    #     "GAT": {
    #         "model": model_gat,
    #         "optimizer": torch.optim.Adam(model_gat.parameters(), lr=lr, weight_decay=weight_decay),
    #         "criterion": torch.nn.CrossEntropyLoss(),
    #     },
    #     "GIN": {
    #         "model": model_gin,
    #         "optimizer": torch.optim.Adam(model_gin.parameters(), lr=lr, weight_decay=weight_decay),
    #         "criterion": torch.nn.CrossEntropyLoss(),
    #     },
    #     "GraphSAGE": {
    #         "model": model_sage,
    #         "optimizer": torch.optim.Adam(model_sage.parameters(), lr=lr, weight_decay=weight_decay),
    #         "criterion": torch.nn.CrossEntropyLoss(),
    #     },  
    # }
    # return bundle
