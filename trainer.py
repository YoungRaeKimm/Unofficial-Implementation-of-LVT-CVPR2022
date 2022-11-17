import torch
import torch.nn as nn
import torch.optim as optim
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
        self.device = config.device
        self.act = nn.Softmax(dim=1)
        if self.dataset == 'tinyimagenet200':
            self.n_classes = 200
        else:
            self.n_classes = 100
        self.increment = self.n_classes/self.split
    
    def train():
        model = LVT(dim=, num_heads=, hidden_dim=, bias=)
        model.to(self.device)
        cross_entropy = nn.CrossEntropyLoss()
        kl_divergence = nn.KLDivLoss()
        optimizr = optim.SGD(modl.parameters(), lr = self.lr)
        memory = torch.zeros(self.memory_size, )

        model.train()
        # Train for each task
        for task in range(self.split):
            data_loader = IncrementalDataLoader(self.dataset, self.data_path, True, self.split, task, self.batch_size)
            # x : (B, 3, 32, 32) | y : (B,) | t : (B,)
            K = self.memory_size // (self.increment * (task+1))

            if task == 0:
                # In task 0, initialize memory
                memory = MemoryDataset(
                    torch.zeros(memory_size, *x.shape[1:]),
                    torch.zeros(memory_size),
                    torch.zeros(memory_size),
                    K
                )
            else:
                memory_loader = DataLoader(MemoryDataset, batch_size=self.batch_size)

            for epoch in range(self.train_epoch):
                # Train current Task
                for x, y, t in data_loader:
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)

                    inj_logit, acc_logit = model(x)
                    L_It = cross_entropy(self.act(inj_logit), y)
                    L_gamma = cross_entropy(self.act(acc_logit), y)
                    
                    inj_loss.backward()
                    optimizer.step()
                
                # Train Examplars from Memory
                if task > 0:
                    for x, y, t in memory_loader:
                        x = x.to(device=self.device)
                        y = y.to(device=self.device)

                        inj_logit, acc_logit = model(x)
                        inj_loss = criterion(inj_logit, y)
                        acc_loss = criterion(acc_logit, y)
                        
                        acc_loss.backward()
                        optimizer.step()

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
                inj_logit, acc_logit = model(x)
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
        