from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, BatchNorm


class GBCNN(nn.Module):
    """Simple graph-based CNN for point-wise edge classification."""

    def __init__(self, input_dim: int = 6):
        super().__init__()
        self.conv1 = GraphConv(input_dim, 64)
        self.bn1 = BatchNorm(64)
        self.conv2 = GraphConv(64, 128)
        self.bn2 = BatchNorm(128)
        self.conv3 = GraphConv(128, 256)
        self.bn3 = BatchNorm(256)

        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(self.bn1(x))
        x = self.conv2(x, edge_index)
        x = F.relu(self.bn2(x))
        x = self.conv3(x, edge_index)
        x = F.relu(self.bn3(x))
        x = self.head(x).squeeze(-1)
        return x
