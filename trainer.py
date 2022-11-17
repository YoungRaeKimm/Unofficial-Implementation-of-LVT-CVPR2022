import torch
import torch.nn as nn
import torch.optim as optim
from models.archs.classifiers import *
from models.archs.lvt import *
from utils import IncrementalDataLoader, confidence_score

class Trainer():
    def __init__(self, config):
        self.dataset = config.dataset
        self.train_epoch = config.epoch
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.split = config.split
        self.memory_size = config.memory_size
        self.ILtype = config.ILtype
        self.data_path = config.data_path
        self.device = config.device
        if self.dataset == 'tinyimagenet200':
            self.n_classes = 200
        else:
            self.n_classes = 100
        self.increment = self.n_classes/self.split
    
    def train():
        model = LVT(dim=, num_heads=, hidden_dim=, bias=)
        model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizr = optim.SGD(modl.parameters(), lr = self.lr)
        memory = torch.zeros(self.memory_size, )

        for epoch in range(self.train_epoch):
            model.train()
            # Train for each task
            for task in range(self.split):
                # Train Current Task
                dataloader = IncrementalDataLoader(self.dataset, self.data_path, True, self.split, task, self.batch_size)
                # x : (B, 3, 32, 32) | y : (B,) | t : (B,)
                
                for x, y, t in dataloader:
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)

                    inj_logit, acc_logit = model(x)
                    inj_loss = criterion(inj_logit, y)
                    acc_loss = criterion(acc_logit, y)
                    
                    inj_loss.backward()
                    optimizer.step()

                # Train Examplars from Memory
                if task == 0:
                    memory = {
                        'x':torch.zeros(self.memory_size, *x.numpy().shape[1:]),
                        'y':torch.zeros(self.memory_size),
                        't':torch.zeros(self.memory_size)
                    }
                else:
                    for i in range(self.memory_size):
                        x, y, t = memory['x'][i], memory['y'][i], memory[]
                    pass
                    # TODO

                # Update Memory
                conf_score_list = []
                x_list = []
                labels_list = []

                # Put x into model & Calculate Confidence Score
                for x, y, t in dataloader:
                    x_list.append(x)
                    labels_list.append(y)
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)
                    inj_logit, acc_logit = model(x)
                    conf_score_list.append(confidence_score(inj_logit, y))
                
                K = self.memory_size // (self.increment * (task+1))

                conf_score = np.array(conf_score_list).flatten()
                labels = torch.cat(labels_list).flatten()
                xs = torch.cat(x_list).reshape(-1, *x.shape[1:])

                # Reduce examplars to K
                if task > 0:
                    K_old = self.memory_size // (self.increment * task)
                    pass

                # Add new examplars
                conf_score_sorted = conf_score.argsort()[:,:,-1]
                for label in range(self.increment*task, self.increment*(task+1)):
                    conf_score_sorted[labels==label][:K]
                
                # TODO
        