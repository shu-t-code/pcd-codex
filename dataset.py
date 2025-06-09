from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph


def read_data_list(list_file: str) -> List[Tuple[str, str]]:
    """Read a text file with lines of `point_path label_path`."""
    pairs = []
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            point_path, label_path = line.split()
            pairs.append((point_path, label_path))
    return pairs


class PointCloudEdgeDataset(Dataset):
    """Dataset for edge classification in point clouds."""

    def __init__(self, samples: List[Tuple[str, str]], k: int = 20):
        self.samples = samples
        self.k = k

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Data:
        """Return a graph data object for the given index.

        If ``label_path`` equals ``point_path``, the file is expected to contain
        seven columns ``[x, y, z, nx, ny, nz, label]``. Otherwise ``point_path``
        should hold six columns ``[x, y, z, nx, ny, nz]`` and ``label_path`` a
        single label column.
        """
        point_path, label_path = self.samples[idx]
        pts = np.load(point_path)

        if label_path == point_path:
            if pts.shape[1] < 7:
                raise ValueError(
                    f"Expected at least 7 columns when point and label paths are the "
                    f"same, got shape {pts.shape}"
                )
            labels = pts[:, 6]
            pts = pts[:, :6]
        else:
            labels = np.load(label_path)

        pts = torch.from_numpy(pts).float()
        labels = torch.from_numpy(labels).view(-1).float()

        edge_index = knn_graph(pts[:, :3], k=self.k, loop=False)
        data = Data(x=pts, edge_index=edge_index, y=labels)
        return data
