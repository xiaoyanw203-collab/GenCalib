"""Dataset loader for latent or volume NPZ test files."""

import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset


def _discover_latent_data_dirs(data_dir, split):
    dirs = []
    if os.path.isdir(data_dir):
        dirs.append(data_dir)
    for sub in (split, "test", "val", "train", "test1"):
        path = os.path.join(data_dir, sub)
        if os.path.isdir(path) and path not in dirs:
            dirs.append(path)
    return dirs


class TestDataset(Dataset):
    """Load test samples from NPZ files containing 'volume' or 'latent' plus 'condition'."""

    def __init__(self, data_dir, split="test1", max_files=None, require_condition=True):
        search_dirs = _discover_latent_data_dirs(data_dir, split)
        self.files = []
        for directory in search_dirs:
            found = sorted(
                f for f in glob(os.path.join(directory, "*.npz")) if split in os.path.basename(f)
            )
            if found:
                self.files = found
                break

        if len(self.files) == 0:
            raise ValueError(f"No .npz files found under {data_dir}")

        if max_files is not None and max_files > 0:
            self.files = self.files[:max_files]

        first_file_data = np.load(self.files[0])
        if "volume" in first_file_data.keys():
            self.data_format = "volume"
            data_key = "volume"
        elif "latent" in first_file_data.keys():
            self.data_format = "latent"
            data_key = "latent"
        else:
            raise ValueError("Data file must contain 'volume' or 'latent' key")
        first_file_data.close()

        self.require_condition = require_condition
        self.file_samples = []
        for file_path in self.files:
            data = np.load(file_path)
            data_array = data[data_key]
            self.file_samples.append(data_array.shape[0] if data_array.ndim == 5 else 1)
            data.close()

        self.cumulative_samples = [0]
        for num_samples in self.file_samples:
            self.cumulative_samples.append(self.cumulative_samples[-1] + num_samples)

        self.total_samples = sum(self.file_samples)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        file_idx = 0
        for i in range(len(self.cumulative_samples) - 1):
            if self.cumulative_samples[i] <= idx < self.cumulative_samples[i + 1]:
                file_idx = i
                break

        sample_idx = idx - self.cumulative_samples[file_idx]
        data = np.load(self.files[file_idx])

        data_key = "volume" if self.data_format == "volume" else "latent"
        gt_data = torch.from_numpy(data[data_key]).float()

        result = {
            "filename": os.path.basename(self.files[file_idx]),
            "data_format": self.data_format,
        }

        if self.require_condition:
            result["condition"] = torch.from_numpy(data["condition"]).float()

        data.close()

        if gt_data.ndim == 5:
            gt_data = gt_data[sample_idx]
        if self.require_condition and result["condition"].ndim == 4:
            result["condition"] = result["condition"][sample_idx]

        result[data_key] = gt_data
        return result
