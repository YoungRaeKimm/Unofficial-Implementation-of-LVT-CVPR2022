
from trainer import Trainer
from config import get_config


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--test', action = 'store_true', default = False, help = 'test')
    parser.add_argument('--log_dir', type = str, default = 'ckpt', help = 'log directory')
    parser.add_argument('--ILtype', type = str, default = 'task', help = 'task, class')
    parser.add_argument('--datapath', type = str, default = '/data/nahappy15/cifar100/', help = 'data path')
    parser.add_argument('--dataset', type = str, default = 'cifar100', help = 'ciar100, tinyimagenet200, imagenet100')
    parser.add_argument('--split', type = int, default = 10, help = 'number of split') # do not change the default value
    parser.add_argument('--alpha', type = float, default = 0.5, help = 'Loss hyperparameter')
    parser.add_argument('--beta', type = float, default = 0.5, help = 'Loss hyperparameter')
    parser.add_argument('--gamma', type = float, default = 0.5, help = 'Loss hyperparameter')
    parser.add_argument('--rt', type = float, default = 1.0, help = 'Loss hyperparameter')
    parser.add_argument('--num_head', type = int, default = 2, help = 'number of attention head')
    parser.add_argument('--hidden_dim', type = int, default = 512, help = 'number of hidden dimension of attention')
    parser.add_argument('--memory_size', type = int, default = 500, help = 'memory buffer size')
    parser.add_argument('--ablate_attn', type = bool, default = False, help = 'ablation option for inter-task attention')
    parser.add_argument('--ablate_memupdate', type = bool, default = False, help = 'ablation option for confidence-aware memory update')
    parser.add_argument('--ablate_inj', type = bool, default = False, help = 'ablation option for injection classifier')
    parser.add_argument('--ablate_acc', type = bool, default = False, help = 'ablation option for accumulation classifier')
    args, _ = parser.parse_known_args()

    config = get_config(dataset=args.dataset)
    ## default
    config.test = args.test
    config.log_dir = args.log_dir
    config.ILtype = args.ILtype
    config.data_path = args.datapath
    config.dataset = args.dataset
    config.split = args.split
    config.alpha = args.alpha
    config.beta = args.beta
    config.gamma = args.gamma
    config.rt = args.rt
    config.num_head = args.num_head
    config.hidden_dim = args.hidden_dim
    config.memory_size = args.memory_size
    config.ablate_attn = args.ablate_attn
    config.ablate_memupdate = args.ablate_memupdate
    config.ablate_inj = args.ablate_inj
    config.ablate_acc = args.ablate_acc


    trainer = Trainer(config)
    if args.test:
        trainer.test()
    else:
        trainer.train()
