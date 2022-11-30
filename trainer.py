import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pickle as pkl
import random
import numpy as np

from torch.utils.data import DataLoader
from copy import deepcopy
from models.archs.classifiers import *
from models.archs.lvt import *
from utils import IncrementalDataLoader, confidence_score, MemoryDataset, toRed, toBlue, toGreen

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
    
transform = [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]

transform_test = [
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]

# Recursively initialize the parameters
def init_xavier(submodule):
    if isinstance(submodule, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(submodule.weight)
        if submodule.bias is not None:
            submodule.bias.data.fill_(0.01)
    elif isinstance(submodule, torch.nn.BatchNorm2d):
        submodule.weight.data.fill_(1.0)
        submodule.bias.data.zero_()
    elif isinstance(submodule, nn.Linear):
        torch.nn.init.xavier_uniform_(submodule.weight)
        if submodule.bias is not None:
            submodule.bias.data.fill_(0.01)
    elif isinstance(submodule, Attention):
        for sm in list(submodule.children()):
            init_xavier(sm)

class Trainer():
    def __init__(self, config):
        self.log_dir = config.log_dir
        self.dataset = config.dataset
        self.train_epoch = config.epoch
        self.batch_size = config.batch_size
        self.lr = config.lr
        self.split = config.split
        self.memory_size = config.memory_size
        self.ILtype = config.ILtype
        self.data_path = config.data_path
        self.scheduler = config.scheduler
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.act = nn.Softmax(dim=1)
        if self.dataset == 'tinyimagenet200':
            self.n_classes = 200
        else:
            self.n_classes = 100
        self.increment = int(self.n_classes//self.split)
        self.resume = config.resume
        self.resume_task = config.resume_task
        self.resume_time = config.resume_time
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
        
        if config.resume:
            self.model = LVT(batch=self.batch_size, n_class=self.increment*self.resume_task, IL_type=self.ILtype, dim=512, num_heads=self.num_head, hidden_dim=self.hidden_dim, bias=self.bias, device=self.device).to(self.device)
            cur_dir = os.path.dirname(os.path.realpath(__file__))
            model_name = f'model_{self.resume_time}_task_{self.resume_task-1}.pt'
            self.model = torch.load(os.path.join(os.path.join(cur_dir, self.log_dir, "saved_models", model_name)), map_location=self.device)
            self.prev_model = deepcopy(self.model).to(self.device)
            self.model.add_classes(self.increment)
        else:
            self.model = LVT(batch=self.batch_size, n_class=self.increment, IL_type=self.ILtype, dim=512, num_heads=self.num_head, hidden_dim=self.hidden_dim, bias=self.bias, device=self.device).to(self.device)
            self.prev_model = None
            self.model.apply(init_xavier)
            
        self.memory = None
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr)
        if self.scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.train_epoch/10, 0.1)
        
    def save(self, model, memory, task):
        model_time = time.strftime("%Y%m%d_%H%M")
        model_name = f"model_{model_time}_task_{task}.pt"
        memory_name = f"memory_{model_time}_task_{task}.pt"
        print(f'Model saved as {model_name}')
        print(f'Memory saved as {memory_name}')
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        print(f'Path : {os.path.join(os.path.join(cur_dir, self.log_dir, "saved_models", model_name))}')
        torch.save(model, os.path.join(os.path.join(cur_dir, self.log_dir, "saved_models", model_name)))
        with open(os.path.join(os.path.join(cur_dir, self.log_dir, "saved_models", memory_name)), 'wb') as f:
            pkl.dump(memory, f)
    
    def train(self):
        cross_entropy = nn.CrossEntropyLoss()
        kl_divergence = nn.KLDivLoss(log_target=True)

        self.model.train()
        # Train for each task
        start_task = self.resume_task if self.resume is True else 0
        for task in range(start_task, self.split):
            grad_saved=False
            data_loader = IncrementalDataLoader(self.dataset, self.data_path, True, self.split, task, self.batch_size, transform)
            # print(data_loader)
            # x : (B, 3, 32, 32) | y : (B,) | t : (B,)
            x = data_loader.dataset[0][0]
            K = self.memory_size // (self.increment * (task+1))

            # memory
            if self.memory is None:
                if self.resume:
                    cur_dir = os.path.dirname(os.path.realpath(__file__))
                    memory_name = f'model_{self.resume_time}_task_{self.resume_task-1}.pt'
                    with open(os.path.join(os.path.join(cur_dir, self.log_dir, "saved_models", memory_name)), 'rb') as f:
                        self.memory = pkl.load(f)
                else:
                    self.memory = MemoryDataset(
                        torch.zeros(self.memory_size, *x.shape),
                        torch.zeros(self.memory_size),
                        torch.zeros(self.memory_size),
                        K
                    )
                # else:
            #     memory_loader = DataLoader(MemoryDataset, batch_size=self.batch_size, shuffle=True)

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
                    # cross_entropy(inj_logit, y).backward()
                    if prev_avg_K_grad is not None:
                        # cross_entropy(y, int_out).backward()
                        prev_avg_K_grad += self.model.get_K_grad()
                        prev_avg_bias_grad += self.model.get_bias_grad()
                    else:
                        # cross_entropy(y, self.act(inj_logit / self.T)).backward()
                        prev_avg_K_grad = self.model.get_K_grad()
                        prev_avg_bias_grad = self.model.get_bias_grad()
                prev_avg_K_grad /= length
                prev_avg_bias_grad /= length
                K_w_prev = self.model.get_K()
                K_bias_prev = self.model.get_bias()



            # train
            for epoch in range(self.train_epoch):
                # Train current Task
                correct, total = 0, 0
                for batch_idx, (x, y, t) in enumerate(data_loader):
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)

                    feature = self.model.forward_backbone(x)
                    inj_logit = self.model.forward_inj(feature)
                    acc_logit = self.model.forward_acc(feature)

                    if task == 0:
                        acc_logit = torch.zeros_like(inj_logit).to(self.device)

                    # print(inj_logit)
                    L_It = cross_entropy(inj_logit, y)
                    L_At = cross_entropy(acc_logit, y)
                    
                    # Train memory if task>0
                    if task > 0:
                        # print(f'prev_K_grad : {prev_avg_K_grad.shape}, K : {self.model.get_K().shape}')
                        # print(f'prev_B_grad : {prev_avg_bias_grad.shape}, B : {self.model.get_bias().shape}')
                        L_a = (torch.abs(torch.tensordot(prev_avg_K_grad, (self.model.get_K() - K_w_prev)))).sum() + \
                                (torch.abs(torch.tensordot(prev_avg_bias_grad, (self.model.get_bias() - K_bias_prev), dims=([2, 1], [2, 1])))).sum()
                        
                        # Train Examplars from Memory
                        
                        # for batch_idx, (x, y, t) in enumerate(memory_loader):
                        # x,y,t = memory_loader.__getitem__(batch_idx%(self.memory_size//self.batch_size))
                        memory_idx = np.random.permutation(self.memory_size)[:self.batch_size]
                        x,y,t = self.memory[memory_idx]
                        # print(x.size())
                        # print(y)
                        x = x.to(self.device)
                        y = y.type(torch.LongTensor).to(self.device)
                        # z = z_list[batch_idx].to(device=self.device)
                        z = self.prev_model.forward_acc(self.model.forward_backbone(x))
                        acc_logit = self.model.forward_acc(self.model.forward_backbone(x))
                        
                        # print(f'For dim: acclogit size {acc_logit.size()}, z size {z.size()}')
                    else:
                        z = acc_logit
                        L_a = torch.zeros_like(L_It).to(self.device)

                    L_r = cross_entropy(acc_logit, y)
                    L_d = kl_divergence(self.act(z/self.T), self.act(acc_logit/self.T))
                    L_l = self.alpha*L_r + self.beta*L_d + self.rt*L_At
                    
                    if task == 0:
                        total_loss = L_It
                    else:
                        total_loss = L_l + L_It + self.gamma*L_a
                    # print(acc_logit.max())
                    _, predicted = torch.max(inj_logit, 1)
                    # print(predicted)
                    # print(y)
                    correct += (predicted == y).sum().item()
                    total += y.size(0)
                    
                    # print(f'batch {batch_idx} | L_l : {L_l}| L_r : {L_r}| L_d : {L_d}| L_At :{L_At}| L_It : {L_It}| L_a : {L_a}| train_loss :{total_loss}|  accuracy : {100*correct/total}')

                    self.optimizer.zero_grad()
                    total_loss.backward()
                    # print(self.model.inj_clf.weight.grad)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # print(f'batch : {batch_idx} | L : {total_loss} | L_It : {L_It} | L_d :{L_d} | acc : {acc_logit.max()}')
                
                print(f'epoch {epoch} | L_l : {L_l:.3f}| L_r : {L_r:.3f}| L_d : {L_d:.3f}| L_At :{L_At:.3f}| L_It : {L_It:.3f}| L_a : {L_a:.3f}| train_loss :{total_loss:.3f} |  accuracy : {100*correct/total:.3f}')
                    
            
            # Save previous model
            self.prev_model = copy.deepcopy(self.model)

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
                feature = self.model.forward_backbone(x)
                inj_logit = self.model.forward_inj(feature)
                conf_score_list.append(confidence_score(inj_logit.detach(), y.detach()).numpy())
                # store logit z=inj_logit for each x
            
            conf_score = np.array(conf_score_list).flatten()
            labels = torch.cat(labels_list).flatten()
            xs = torch.cat(x_list).view(-1, *x.shape[1:])

            # Reduce examplars to K
            if task > 0:
                self.memory.remove_examplars(K)

            # Add new examplars
            conf_score_sorted = conf_score.argsort()[::-1]
            for label in range(self.increment*task, self.increment*(task+1)):
                new_x = xs[conf_score_sorted[labels==label][:K]]
                new_y = labels[conf_score_sorted[labels==label][:K]]
                new_t = torch.full((K,), task)
                self.memory.update_memory(label, new_x, new_y, new_t)
                
            # updatae r(t)
            self.rt *= 0.9

            if self.ILtype == 'class':
                self.model.add_class(self.increment)
                self.cur_classes += self.increment
            
            self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr)
            if self.scheduler:
                self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.train_epoch/10, 0.1)
                
            self.save(self.model, self.memory, task)
    
    def eval(self, task):
        self.model.eval()
        data_loader = IncrementalDataLoader(self.dataset, self.data_path, False, self.split, task, self.batch_size, transform_test)
        correct, total = 0, 0
        for x, y, _ in data_loader:
            x = x.to(device=self.device)

            acc_logit = self.model.forward_acc(self.model.forward_backbone(x))
            _, predicted = torch.max(acc_logit, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)
            
        print(toGreen(f'Test accuracy on task {task} : {100*correct/total}'))
        self.model.train()