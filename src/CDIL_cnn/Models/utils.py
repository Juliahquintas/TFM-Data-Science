"""
utils.py — Dataset y semilla aleatoria (exactos de Marta Rey).
"""
import os, random
import torch, numpy as np
from torch.utils.data import Dataset


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class DatasetCreator(Dataset):
    def __init__(self, data, labels):
        self.data, self.labels = data, labels
    def __getitem__(self, i):
        return self.data[i], self.labels[i]
    def __len__(self):
        return len(self.labels)
