import numpy as np
import torch
from torch.utils.data import Dataset


class SequenceDataset(Dataset):
    """Simple dataset of sequences and targets."""

    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        # sequences: (N, seq_len, features)
        self.X = sequences.astype(np.float32)
        self.y = targets.astype(np.long)  # For classification

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


def create_sequences(values, seq_len: int):
    """Create (sequences, target) from a 1D or 2D array of values.

    values shape: (T, features) or (T,)
    returns sequences shape: (N, seq_len, features), targets shape: (N, features)
    """
    arr = np.array(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)

    seqs = []
    targets = []
    for i in range(len(arr) - seq_len):
        seqs.append(arr[i: i + seq_len])
        targets.append(arr[i + seq_len])

    return np.stack(seqs), np.stack(targets)
