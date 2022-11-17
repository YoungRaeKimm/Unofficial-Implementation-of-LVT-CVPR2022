import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.archs.classifiers import *
from models.archs.lvt import *
from utils import IncrementalDataLoader, confidence_score, MemoryDataset

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
        self.scheduler = config.sheduler
        self.device = torch.device('cuda')
        self.act = nn.Softmax(dim=1)
        if self.dataset == 'tinyimagenet200':
            self.n_classes = 200
        else:
            self.n_classes = 100
        self.increment = self.n_classes/self.split
        
        # hyper parameter
        self.num_head = 2
        self.hidden_dim = 512
        self.bias = True
        
        self.model = LVT(dim=512, num_heads=self.num_head, hidden_dim=self.hidden_dim, bias=self.bias)
        self.model.to(self.device)
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr)
        if self.scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.train_epoch/10, 0.1)
        
    
    def train(self):
        cross_entropy = nn.CrossEntropyLoss()
        kl_divergence = nn.KLDivLoss()
        memory = torch.zeros(self.memory_size, )

        self.model.train()
        # Train for each task
        for task in range(self.split):
            data_loader = IncrementalDataLoader(self.dataset, self.data_path, True, self.split, task, self.batch_size)
            # x : (B, 3, 32, 32) | y : (B,) | t : (B,)
            K = self.memory_size // (self.increment * (task+1))

            if task == 0:
                # In task 0, initialize memory
                memory = MemoryDataset(
                    torch.zeros(self.memory_size, *x.shape[1:]),
                    torch.zeros(self.memory_size),
                    torch.zeros(self.memory_size),
                    K
                )
            else:
                memory_loader = DataLoader(MemoryDataset, batch_size=self.batch_size)

            for epoch in range(self.train_epoch):
                # Train current Task
                for x, y, t in data_loader:
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)

                    feature = self.model.forward_backbone(x)
                    inj_logit = self.model.forward_inj(feature)
                    L_It = cross_entropy(self.act(self.model.forward_inj(x)), y)
                    L_a = None
                    
                    
                    
                # Train Examplars from Memory
                if task > 0:
                    for x, y, t in memory_loader:
                        x = x.to(device=self.device)
                        y = y.to(device=self.device)

                        acc_logit = self.model.forward_acc(self.model.forward_backbone(x))
                        L_r = cross_entropy(acc_logit, y)
                        L_d = None 
                        L_l = None
                        
                # accumulate losses at the end of epoch? or end of iteration?        
                        
                acc_loss.backward()
                self.optimizer.step()

            # Update Memory
            conf_score_list = []
            x_list = []
            labels_list = []
            # Calculate Confidence Score
            for x, y, t in data_loader:
                x_list.append(x)
                labels_list.append(y)
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                inj_logit, acc_logit = self.model(x)
                conf_score_list.append(confidence_score(inj_logit, y))
            
            conf_score = np.array(conf_score_list).flatten()
            labels = torch.cat(labels_list).flatten()
            xs = torch.cat(x_list).reshape(-1, *x.shape[1:])

            # Reduce examplars to K
            if task > 0:
                memory.remove_examplars(K)

            # Add new examplars
            conf_score_sorted = conf_score.argsort()[:,:,-1]
            for label in range(self.increment*task, self.increment*(task+1)):
                new_x = xs[conf_score_sorted[labels==label][:K]]
                new_y = labels[conf_score_sorted[labels==label][:K]]
                new_t = torch.full((K,), task)
                memory.update_memory(label, new_x, new_y, new_t)
        