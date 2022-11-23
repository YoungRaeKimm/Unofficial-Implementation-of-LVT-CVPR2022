import torch
from torch.utils.data import DataLoader, Dataset
from continuum import ClassIncremental
from continuum.datasets import CIFAR100, TinyImageNet200, ImageNet100
import numpy as np

def IncrementalDataLoader(dataset_name, data_path, train, n_split, task_id, batch_size, transform):
    if task_id >= n_split:
        return False

    dataset_name = dataset_name.lower()
    n_classes = 100
    if dataset_name == 'cifar100':
        dataset = CIFAR100(data_path, download=False, train=train, transform=transform)
    if dataset_name == 'tinyimagenet200':
        dataset = TinyImageNet200(data_path, download=False, train=train, transform=transform)
        n_classes = 200
    if dataset_name == 'imagenet100':
        dataset = ImageNet100(data_path, download=False, train=train, transform=transform)
    else:
        return False

    scenario = ClassIncremental(dataset, increment=n_classes//n_split)
    loader = DataLoader(scenario[task_id], batch_size = batch_size, shuffle=True)
    return loader

def confidence_score(z, c):
    B = z.shape[0]
    score = np.zeros(B)
    for i in range(B):
        score[i] = np.exp(z[i, c[i]]) / (np.exp(z[i, :])).sum()
    return score

class MemoryDataset(Dataset):
    def __init__(self, x, y, t, k):
        self.x = x
        self.y = y
        self.t = t
        self.k = k
        self.size = len(self.x)
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.x, self.y, self.t

    def remove_examplars(self, new_k):
        new_x = torch.zeros_like(self.x)
        new_y = torch.zeros_like(self.y)
        new_t = torch.zeros_like(self.t)

        for i, start in enumerate(range(0, self.size, self.k)):
            new_x[new_k*i:new_k*(i+1)] = self.x[start:start+new_k]
            new_y[new_k*i:new_k*(i+1)] = self.y[start:start+new_k]
            new_t[new_k*i:new_k*(i+1)] = self.t[start:start+new_k]

        self.x = new_x
        self.y = new_y
        self.t = new_t
        self.k = new_k
    
    def update_memory(self, label, new_x, new_y, new_t):
        self.x[label*self.k:(label+1)*self.k] = new_x
        self.y[label*self.k:(label+1)*self.k] = new_y
        self.t[label*self.k:(label+1)*self.k] = new_t