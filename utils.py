import os
import h5py
import numpy as np
import deepdish as dd
import pandas as pd
from scipy.spatial.distance import cdist
from torch.utils.data import Dataset, DataLoader


class read_files:

    def __init__(self):
        self.filename = "data.xxx"

    def read_f5(self):
        h5f = dd.io.load(self.filename)
        # h5f = h5py.File(self.filename,"r")
        return list(h5f.keys()), list(h5f.values())

    def read_csv(self):
        data = pd.read_csv(self.filename)
        return data.values

    def read_npz(self):
        data = np.load(self.filename)
        return data


class MyDataset(Dataset):
    ##  to be used later
    def __init__(self, x, y, key, val_len, test_len, transform=None):
        self.transform = transform
        self.data = x
        self.y = y
        self.key = key
        self._len = {
            "train_len": x.shape[0] - val_len - test_len,
            "validate_len": val_len,
            "test_len": test_len
        }

    def __getitem__(self, item):
        if self.key == "train":
            return self.x[item], self.y[item]
        elif self.key == "validate":
            return self.x[self._len["train_len"] + item], self.y[self._len["train_len"] + item]
        elif self.key == "test":
            return self.x[-self._len["test_len"] + item], self.y[-self._len["test_len"] + item]
        else:
            raise NotImplementedError()

    def __len__(self):
        return self._len[f"{self.key}_len"]


class ReadDataset(Dataset):
    def __init__(self, x, y):
        self.feature = x
        self.label = y

    def __getitem__(self, idx):
        if self.label is None:
            print(" there is no label")
            return self.feature[idx]
        return self.feature[idx], self.label[idx]

    def __len__(self):
        return len(self.label)


def minmaxscaler(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def standardscaler(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def dividemaxscaler(data):
    return data/np.max(data)


def unnormalize(data, data_std, data_mean):
    return data * data_std + data_mean


def cal_adjacent_matrix(data, normalized_category="laplacian"):
    hidden_size = 40
    print(" svd decomposition")
    u, s, v = np.linalg.svd(np.array(data))
    print(" to get the station representation")

    w = np.diag(s[:hidden_size]).dot(v[:hidden_size, :]).T
    print("calculate the distance between stations")
    graph = cdist(w, w, metric='euclidean')
    print(" use a Gaussian methond to transfer the distance to weights between stations")
    a = graph * -1 / np.std(graph) ** 2
    support = np.exp(a)
    support = support - np.identity(
        support.shape[0])  # np.identity: creat a matrix (M,M) in which the main diagonal is 1, the rest is 0
    if normalized_category == 'randomwalk':
        support = random_walk_matrix(support)
    elif normalized_category == 'laplacian':
        support = normalized_laplacian(support)
    return support


def random_walk_matrix(w) -> np.matrix:
    d = np.array(w.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = np.eye(d_inv.shape[0]) * d_inv
    return d_mat_inv.dot(w)


def normalized_laplacian(w: np.ndarray) -> np.matrix:
    d = np.array(w.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    print(d, d_inv_sqrt)
    d_mat_inv_sqrt = np.eye(d_inv_sqrt.shape[0]) * d_inv_sqrt.shape
    return np.identity(w.shape[0]) - d_mat_inv_sqrt.dot(w).dot(d_mat_inv_sqrt)
