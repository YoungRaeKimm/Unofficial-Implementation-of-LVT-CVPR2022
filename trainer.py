import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from models.archs.classifiers import *
from models.archs.lvt import *
from utils import IncrementalDataLoader, confidence_score, MemoryDataset

transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

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
        self.increment = int(self.n_classes//self.split)
        self.cur_classes = self.increment
        
        # hyper parameter
        self.num_head = 2
        self.hidden_dim = 512
        self.bias = True
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.rt = 1.
        self.T = 2. # softmax temperature
        # TODO : Monotonically Decreasing Function gamma(t)
        
        self.model = LVT(n_class=self.n_classes, dim=512, num_heads=self.num_head, hidden_dim=self.hidden_dim, bias=self.bias).to(self.device)
        self.prev_model = None
        if self.ILtype == 'task':
            self.classifiers = []
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr)
        if self.scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.train_epoch/10, 0.1)
        
    
    def train(self):
        cross_entropy = nn.CrossEntropyLoss()
        kl_divergence = nn.KLDivLoss()

        self.model.train()
        # Train for each task
        for task in range(self.split):
            grad_saved=False
            data_loader = IncrementalDataLoader(self.dataset, self.data_path, True, self.split, task, self.batch_size, transform)
            # x : (B, 3, 32, 32) | y : (B,) | t : (B,)
            K = self.memory_size // (self.increment * (task+1))

            # memory
            if task == 0:
                # In task 0, initialize memory
                memory = MemoryDataset(
                    torch.zeros(self.memory_size, *x.shape[1:]),
                    torch.zeros(self.memory_size),
                    torch.zeros(self.memory_size),
                    K
                )
            else:
                memory_loader = DataLoader(MemoryDataset, batch_size=self.batch_size, shuffle=True)

            # average gradient
            if task > 0:
                prev_avg_K_grad = None
                prev_avg_bias_grad = None
                length = 0
                for x, y, _ in data_loader:
                    length += 1
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)

                    inj_logit = self.model.forward_inj(self.model.forward_backbone(x))
                    if prev_avg_K_grad is not None:
                        cross_entropy(y, self.act(inj_logit / self.T)).backward()
                        prev_avg_K_grad += self.model.get_K_grad()
                        prev_avg_bias_grad += self.model.get_bias_grad()
                    else:
                        cross_entropy(y, self.act(inj_logit / self.T)).backward()
                        prev_avg_K_grad = self.model.get_K_grad()
                        prev_avg_bias_grad = self.model.get_bias_grad()
                prev_avg_K_grad /= length
                prev_avg_bias_grad /= length
                K_w_prev = self.model.get_K()
                K_bias_prev = self.model.get_bias()


            # train
            for epoch in range(self.train_epoch):
                # Train current Task
                for batch_idx, (x, y, t) in enumerate(data_loader):
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)

                    feature = self.model.forward_backbone(x)
                    inj_logit = self.model.forward_inj(feature)
                    acc_logit = self.model.forward_acc(feature)
                    L_It = cross_entropy(y, self.act(inj_logit))
                    L_At = cross_entropy(y, acc_logit)
                    
                    L_a = None
                    if task > 0:
                        L_a = (torch.abs(torch.tensordot(prev_avg_K_grad, (self.model.get_K() - K_w_prev)))).sum() + \
                                (torch.abs(torch.tensordot(prev_avg_bias_grad, (self.model.get_bias() - K_bias_prev)))).sum()
                        
                        # Train Examplars from Memory
                        
                        # for batch_idx, (x, y, t) in enumerate(memory_loader):
                        x,y,t = memory_loader.__getitem__(batch_idx%(self.memory_size//self.batch_size))
                        x = x.to(device=self.device)
                        y = y.to(device=self.device)
                        # z = z_list[batch_idx].to(device=self.device)
                        z = self.prev_model.forward_acc(self.model.forward_backbone(x))
                        acc_logit = self.model.forward_acc(self.model.forward_backbone(x))
                        
                        print(f'For dim: acclogit size {acc_logit.size()}, z size {z.size()}')

                        L_r = self.act(cross_entropy(y, acc_logit))
                        L_d = kl_divergence(self.act(z/self.T), acc_logit)
                        L_l = self.alpha*L_r + self.beta*L_d + self.rt*L_At
                        
                    total_loss = L_l + L_It + self.gamma*L_a
                    total_loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
            
            # Save previous model
            self.prev_model = self.model.clone().detach()

            # Update Memory
            conf_score_list = []
            x_list = []
            labels_list = []
            # z_list = []
            # Calculate Confidence Score
            for x, y, t in data_loader:
                x_list.append(x)
                labels_list.append(y)
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                feature = self.model.forward_backbone(x)
                inj_logit = self.model.forward_inj(feature)
                acc_logit = self.model.forward_acc(feature)
                conf_score_list.append(confidence_score(inj_logit, y))
                # store logit z=inj_logit for each x
                # z_list.append(acc_logit)
            
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
                
            # updatae r(t)
            self.rt *= 0.9

            if self.ILtype == 'class':
                self.model.add_class(self.increment)
                self.cur_classes += self.increment