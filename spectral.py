from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import numpy as np

class Spectral:
    def __init__(self, X_fn, y_fn, batch_size=10, shuffle=True, num_workers=4, min_idx=None, 
                 max_idx=None, sampler=None, X_fn_v=None, y_fn_v=None, X_fn_t=None, y_fn_t=None):
        
        self.train = None
        self.val = None
        self.test = None

        self.train = spectral_dataloader(X_fn, y_fn,
        batch_size=batch_size, shuffle=True, num_workers=2)

        if X_fn_v is not None and y_fn_v is not None:
            self.val = spectral_dataloader(X_fn_v, y_fn_v,
            batch_size=batch_size, shuffle=False, num_workers=2)
        
        if X_fn_t is not None and y_fn_t is not None:
            self.test = spectral_dataloader(X_fn_t, y_fn_t,
            batch_size=batch_size, shuffle=False, num_workers=2)
        
        if self.val is None and self.test is not None:
            print("Splitting Train into 90/10 Train, Val Split")
            p_val = 0.1
            n_val = int(len(X_fn) * p_val)
            idx_tr = list(range(len(X_fn)))
            np.random.shuffle(idx_tr)
            idx_val = idx_tr[:n_val]
            idx_tr = idx_tr[n_val:]
    
            self.train = spectral_dataloader(X_fn, y_fn, idxs=idx_tr,
            batch_size=batch_size, shuffle=True, num_workers=2)
            self.val = spectral_dataloader(X_fn, y_fn, idxs=idx_val,
            batch_size=batch_size, shuffle=False, num_workers=2)


        if self.val is None and self.test is None:
            print("Splitting Train into Random 80/10/10 Train, Val, Test Split")
            X = np.load(X_fn)
            y = np.load(y_fn)
            p_val = 0.1
            n_val = int(len(X) * p_val)
            idx_tr = list(range(len(X)))
            np.random.shuffle(idx_tr)
            idx_tr = idx_tr[n_val*2:]
            idx_val = idx_tr[n_val:n_val*2]
            idx_te = idx_tr[:n_val]
            
            self.train = spectral_dataloader(X, y, idxs=idx_tr,
            batch_size=batch_size, shuffle=True, num_workers=2)
            self.val = spectral_dataloader(X, y, idxs=idx_val,
            batch_size=batch_size, shuffle=False, num_workers=2)
            self.test = spectral_dataloader(X, y, idxs=idx_te,
            batch_size=batch_size, shuffle=False, num_workers=2)


class SpectralDataset(Dataset):
    """
    Builds a dataset of spectral data. Use idxs to specify which samples to use
    for dataset - this allows for random splitting into training, validation,
    and test sets. Instead of passing in filenames for X and y, we can also
    pass in numpy arrays directly.
    """
    def __init__(self, X_fn, y_fn, idxs=None, transform=None):
        if type(X_fn) == str:
            self.X = np.load(X_fn)
        else:
            self.X = X_fn
        if type(y_fn) == str:
            self.y = np.load(y_fn)
        else:
            self.y = y_fn
        if idxs is None: idxs = np.arange(len(self.y))
        self.idxs = idxs
        self.transform = transform

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        i = self.idxs[idx]
        x, y = self.X[i], self.y[i]
        x = np.expand_dims(x, axis=0)
        if self.transform:
            x = self.transform(x)
        return (x, y)

class Spectral_30_class:
    def __init__(self, X_train, y_train, batch_size=16, shuffle=True, num_workers=4, min_idx=None, 
                 max_idx=None, sampler=None, X_val=None, y_val=None, X_test=None, y_test=None):
        
        self.train = None
        self.val = None
        self.test = None

        self.train = spectral_dataloader(X_train, y_train,
        batch_size=batch_size, shuffle=shuffle, num_workers=2)

        self.val = spectral_dataloader(X_val, y_val,
        batch_size=batch_size, shuffle=False, num_workers=2)

        self.test = spectral_dataloader(X_test, y_test,
        batch_size=batch_size, shuffle=False, num_workers=2)

### TRANSFORMS ###


class GetInterval(object):
    """
    Gets an interval of each spectrum.
    """
    def __init__(self, min_idx, max_idx):
        self.min_idx = min_idx
        self.max_idx = max_idx

    def __call__(self, x):
        x = x[:,self.min_idx:self.max_idx]
        return x


class ToFloatTensor(object):
    """
    Converts numpy arrays to float Variables in Pytorch.
    """
    def __call__(self, x):
        x = torch.from_numpy(x).float()
        return x


### TRANSFORMS ###


def spectral_dataloader(X_fn, y_fn, idxs=None, batch_size=10, shuffle=True,
    num_workers=4, min_idx=None, max_idx=None, sampler=None):
    """
    Returns a DataLoader with spectral data.
    """
    transform_list = []
    if min_idx is not None and max_idx is not None:
        transform_list.append(GetInterval(min_idx, max_idx))
    transform_list.append(ToFloatTensor())
    transform = transforms.Compose(transform_list)
    dataset = SpectralDataset(X_fn, y_fn, idxs=idxs, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, sampler=sampler)
    return dataloader


def spectral_dataloaders(X_fn, y_fn, n_train=None, p_train=0.8, p_val=0.1,
    n_test=None, batch_size=10, shuffle=True, num_workers=4, min_idx=None,
    max_idx=None):
    """
    Returns train, val, and test DataLoaders by splitting the dataset randomly.
    Can also take X_fn and y_fn as numpy arrays.
    """
    if type(y_fn) == str:
        idxs = np.arange(len(np.load(y_fn)))
    else:
        idxs = np.arange(len(y_fn))
    np.random.shuffle(idxs)
    if n_train is None: n_train = int(p_train * len(idxs))
    n_val = int(p_val * n_train)
    val_idxs, train_idxs = idxs[:n_val], idxs[n_val:n_train]
    if n_test is None: test_idxs = idxs[n_train:]
    else: test_idxs = idxs[n_train:n_train+n_test]
    trainloader = spectral_dataloader(X_fn, y_fn, train_idxs,
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        min_idx=min_idx, max_idx=max_idx)
    valloader = spectral_dataloader(X_fn, y_fn, val_idxs,
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        min_idx=min_idx, max_idx=max_idx)
    testloader = spectral_dataloader(X_fn, y_fn, test_idxs,
        batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
        min_idx=min_idx, max_idx=max_idx)
    return (trainloader, valloader, testloader)
