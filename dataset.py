from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class RamanSpectra:
    def __init__(self, spectra_dir, label_dir, spectra_interval, seed=42, shuffle=True, num_workers=2, batch_size=16, transform=None):
        assert len(spectra_dir) == len(label_dir) == len(spectra_interval), 'Spectra, labels, and invervals do not align.'
        np.random.seed(seed)

        X_train, X_val, X_test, y_train, y_val, y_test = self._get_splits(spectra_dir, label_dir, spectra_interval)
        
        train_dataset = RamanSpectraDataset(X_train, y_train, transform=ToFloatTensor())
        val_dataset = RamanSpectraDataset(X_val, y_val, transform=ToFloatTensor())
        test_dataset = RamanSpectraDataset(X_test, y_test, transform=ToFloatTensor())

        self.train = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
        self.val = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
        self.test = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    def _get_splits(self, spectra_dir, label_dir, spectra_interval):
        X_train, X_val, X_test, y_train, y_val, y_test = [], [], [], [], [], []

        for spectra, label, interval in zip(spectra_dir, label_dir, spectra_interval):
            X = np.load(spectra)
            y = np.load(label)

            for i, l in enumerate(np.unique(y)):
                X_type = X[y==l]
                num_idxs = len(X_type) // interval
                split = np.arange(num_idxs)
                np.random.shuffle(split)

                for num in split[:-2]:
                    X_train.extend(X_type[num*interval:num*interval+interval])
                    y_train.extend([i]*interval)

                val_num = split[-2]
                X_val.extend(X_type[val_num*interval:val_num*interval+interval])
                y_val.extend([i]*interval)

                test_num = split[-1]
                X_test.extend(X_type[test_num*interval:test_num*interval+interval])
                y_test.extend([i]*interval)

        return X_train, X_val, X_test, y_train, y_val, y_test

class RamanSpectraDataset(Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X, y = self.X[idx], self.y[idx]
        X = np.expand_dims(X, axis=0)
        
        if self.transform:
            X = self.transform(X)

        return (X, y)

class ToFloatTensor(object):
    def __call__(self, x):
        x = torch.from_numpy(x).float()
        return x
