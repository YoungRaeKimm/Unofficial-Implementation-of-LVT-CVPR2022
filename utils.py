from torch.utils.data import DataLoader
from continuum import ClassIncremental
from continuum.datasets import CIFAR100, TinyImageNet200, ImageNet100

def IncrementalDataLoader(dataset_name, data_path, train, n_split, task_id):
    if task_id >= n_split:
        return False

    dataset_name = dataset_name.lower()
    n_classes = 100
    if dataset_name == 'cifar100':
        dataset = CIFAR100(data_path, download=False, train=train)
    if dataset_name == 'tinyimagenet200':
        dataset = TinyImageNet200(data_path, download=False, train=train)
        n_classes = 200
    if dataset_name == 'imagenet100':
        dataset = ImageNet100(data_path, download=False, train=train)
    else:
        return False

    scenario = ClassIncremental(dataset, increment=n_classes//n_split)
    loader = DataLoader(scenario[task_id])
    return loader