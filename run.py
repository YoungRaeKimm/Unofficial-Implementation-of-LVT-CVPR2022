
import torch

from trainer import Trainer
from config import get_config


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--ILtype', type = str, default = 'task', help = 'task, class')
    parser.add_argument('--dataset', type = str, default = 'cifar100', help = 'ciar100, tinyimagenet200, imagenet100')
    parser.add_argument('--split', type = int, default = 10, help = 'number of split') # do not change the default value
    parser.add_argument('--LRS', action = 'store_true', default = False, help = 'whether to use scheduler')
    parser.add_argument('--memorysize', type = int, default = 200, help = '200 or 500')
    parser.add_argument('--everytest', action = 'store_true', default = False, help = 'whether to test at every epoch')
    args, _ = parser.parse_known_args()

    config = get_config(dataset=args.dataset)
    ## default
    config.ILtype = args.ILtype
    config.dataset = args.dataset
    config.split = args.split
    config.LRS = args.LRS
    config.memory_size = args.memorysize
    config.test_every_epoch = args.everytest


    trainer = Trainer(config)
    trainer.train()
