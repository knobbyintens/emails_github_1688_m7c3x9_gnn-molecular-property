"""Clinical feature preprocessing."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ClinicalDataset(Dataset):
    def __init__(self, path: str, feature_names: list[str] | None = None) -> None:
        payload = np.load(Path(path))
        self.features = torch.tensor(payload["features"], dtype=torch.float32)
        self.labels = torch.tensor(payload["labels"], dtype=torch.long)
        self.feature_names = feature_names or [f"feature_{i}" for i in range(self.features.size(1))]

    @property
    def num_features(self) -> int:
        return self.features.size(1)

    def __len__(self) -> int:
        return self.labels.size(0)

    def __getitem__(self, index: int):
        return self.features[index], self.labels[index]
