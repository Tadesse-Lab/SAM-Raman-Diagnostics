import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List
from collections import defaultdict


class SAMRamanDataset(Dataset):
    def __init__(
        self,
        spectral_data: List[np.ndarray],
        labels: List[np.ndarray],
    ):
        self.spectral_data = spectral_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data = torch.tensor(self.spectral_data[idx], dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)

        return data, label


class SAMRaman:
    def __init__(
        self,
        spectral_dirs: List[str],
        label_dirs: List[str],
        patient_intervals: List[int],
        batch_size: int = 16,
        seed: int = 42,
    ):
        self.spectral_dirs = spectral_dirs
        self.label_dirs = label_dirs
        self.patient_intervals = patient_intervals
        self.batch_size = batch_size
        np.random.seed(seed)

        self.label_map = self._get_label_mapping()
        self.inverse_map = {idx: l for l, idx in self.label_map.items()}
        self.train_set, self.val_set, self.test_set = self._get_splits()
        self._create_dataloaders()

    def _get_label_mapping(self):
        labels = []
        for f in self.label_dirs:
            labels.extend(np.load(f))

        label_map = {l: idx for idx, l in enumerate(np.unique(labels))}

        return label_map

    def _get_splits(self):
        train_set, val_set, test_set = (
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
        )

        for spectral_path, label_path, p_int in zip(
            self.spectral_dirs, self.label_dirs, self.patient_intervals
        ):
            spectra = np.load(spectral_path)
            labels = np.load(label_path)

            for l in np.unique(labels):
                label_spectra = spectra[labels == l]
                num_patients = len(label_spectra) // p_int
                patient_idxs = np.arange(num_patients)
                patient_idxs = np.random.permutation(patient_idxs)

                for i in patient_idxs[:-2]:
                    train_set["data"].extend(
                        label_spectra[i * p_int : i * p_int + p_int]
                    )
                    train_set["labels"].extend([self.label_map[l]] * p_int)

                val_i = patient_idxs[-2]
                val_set["data"].extend(
                    label_spectra[val_i * p_int : val_i * p_int + p_int]
                )
                val_set["labels"].extend([self.label_map[l]] * p_int)

                test_i = patient_idxs[-1]
                test_set["data"].extend(
                    label_spectra[test_i * p_int : test_i * p_int + p_int]
                )
                test_set["labels"].extend([self.label_map[l]] * p_int)

        return train_set, val_set, test_set

    def _create_dataloaders(self):
        train_dataset = SAMRamanDataset(
            self.train_set["data"], self.train_set["labels"]
        )
        val_dataset = SAMRamanDataset(self.val_set["data"], self.val_set["labels"])
        test_dataset = SAMRamanDataset(self.test_set["data"], self.test_set["labels"])

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, batch_size=self.batch_size, shuffle=False
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )

    def _get_original_label(self, pred_idx):
        return self.inverse_map[pred_idx]
