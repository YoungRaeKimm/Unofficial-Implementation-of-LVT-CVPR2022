from easydict import EasyDict as edict


def get_config(dataset = 'cifar100'):
    ## GLOBAL
    config = edict()

    if dataset == 'cifar100':
        config.batch_size = 32
        config.epoch = 100
        config.lr=0.01
        config.split=10
        config.memory_size = 500
        config.ILtype = 'task'
        config.scheduler = False

    elif dataset == 'tinyimagenet200':
        config.batch_size = 32
        config.epoch = 100
        config.lr=0.1
        config.split=10
        config.memory_size = 200
        config.ILtype = 'task'
        config.scheduler = False

    elif dataset == 'imagenet100':
        config.batch_size = 32
        config.epoch = 100
        config.lr=0.1
        config.split=10
        config.memory_size = 200
        config.ILtype = 'task'
        config.scheduler = True

    return config
