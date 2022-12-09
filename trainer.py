import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle as pkl
import random
import numpy as np
import logging

from copy import deepcopy
from models.lvt import *
from utils import IncrementalDataLoader, confidence_score, MemoryDataset, get_transforms, toRed, toBlue, toGreen
    

'''random seed'''
seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
        
'''Recursively initialize the parameters'''
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

'''
All training and testing functions are implemented in this class.
'''
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
        self.cur_classes = self.increment
        self.model_time = time.strftime("%Y%m%d_%H%M%S")
        self.ablate_attn = config.ablate_attn
        self.ablate_memupdate = config.ablate_memupdate
        self.ablate_inj = config.ablate_inj
        self.ablate_acc = config.ablate_acc
        assert(np.array([self.ablate_attn, self.ablate_memupdate, self.ablate_inj, self.ablate_acc]).sum() <= 1)
        
        # hyper parameter
        self.num_head = config.num_head             # number of heads in attention 
        self.hidden_dim = config.hidden_dim         # number of hidden dimension in attention
        self.bias = True                            # bias in Transformer or shrink module
        self.alpha = config.alpha                   # coefficient of L_r
        self.beta = config.beta                     # coefficient of L_d
        self.gamma = config.gamma                   # coefficient of L_a
        self.rt = config.rt                         # coefficient of L_At
        self.T = 2.                                 # softmax temperature, which is used in distillation loss
        
        '''
        Create the LVT and initialize the parameters.
        '''
        self.model = LVT(batch=self.batch_size, n_class=self.increment, IL_type=self.ILtype, dim=512, num_heads=self.num_head, hidden_dim=self.hidden_dim, bias=self.bias, device=self.device, ablation = self.ablate_attn).to(self.device)
        self.prev_model = None
        self.model.apply(init_xavier)
        
        '''
        Since dimension of memory depends on the dimension of input image,
        Initialize them on train phase.
        '''
        self.memory = None
        self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr)
        if self.scheduler:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.train_epoch/10, 0.1)
        
        '''random seed'''
        seed = 1234
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        '''
        Set logger and log configs
        '''
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_name = f"{self.model_time}_ablation.log"
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        if os.path.exists(os.path.join(cur_dir, self.log_dir)) == False:
            os.makedirs(os.path.join(cur_dir, self.log_dir,'logs'), exist_ok=True)
            os.makedirs(os.path.join(cur_dir, self.log_dir,'saved_models'), exist_ok=True)
            os.makedirs(os.path.join(cur_dir, self.log_dir,'best_models'), exist_ok=True)
        file_handler = logging.FileHandler(os.path.join(cur_dir, self.log_dir, 'logs', log_name))
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.info(f'alpha :{self.alpha} | beta : {self.beta} | gamma : {self.gamma} | rt : {self.rt} | num_head : {self.num_head} | hidden_dim : {self.hidden_dim} | memory_size : {self.memory_size} | dataset : {self.dataset}')
        ablated = np.array(['attn', 'acc', 'inj', 'memory update'])[np.array([self.ablate_attn, self.ablate_acc, self.ablate_inj, self.ablate_memupdate])]
        self.logger.info(f'ablated : {ablated[0]}')
        
    '''
    Save the model according to the task number.
    '''
    def save(self, model, task):
        model_name = f"{self.dataset}_model_{self.model_time}_task_{task}.pt"
        print(f'Model saved as {model_name}')
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        print(f'Path : {os.path.join(os.path.join(cur_dir, self.log_dir, "saved_models", model_name))}')
        self.logger.info(f'Model saved as {model_name}')
        torch.save(model, os.path.join(os.path.join(cur_dir, self.log_dir, "saved_models", model_name)))


    '''
    Core function.
    This function trains the model during whole tasks.
    '''
    def train(self):
        '''
        We use cross entropy loss for getting classification loss
        and KL divergence loss to distillate the knowledge of previous task model.
        '''
        cross_entropy = nn.CrossEntropyLoss()
        kl_divergence = nn.KLDivLoss(reduction='batchmean')

        self.model.train()
        '''
        Task starts.
        '''
        for task in range(self.split):
            data_loader = IncrementalDataLoader(self.dataset, self.data_path, True, self.split, task, self.batch_size, get_transforms(self.dataset))
            # x : (B, 3, 32, 32) | y : (B,) | t : (B,)
            x = data_loader.dataset[0][0]
            K = self.memory_size // (self.increment * (task+1))

            '''
            Initialize memory buffer.
            '''
            if self.memory is None:
                self.memory = MemoryDataset(
                    torch.zeros(self.memory_size, *x.shape),
                    torch.zeros(self.memory_size),
                    torch.zeros(self.memory_size),
                    torch.zeros(self.memory_size, self.increment),
                    K
                )

            
            '''
            In LVT paper, the authors said that the gradient values of key and bias of attention module 
            represents the importance the last task. (equation (2))
            The average value of gradient is calculated in here.
            '''
            if task > 0 and not self.ablate_attn:
                prev_avg_K_grad = None
                prev_avg_bias_grad = None
                length = 0
                prev_data_loader = IncrementalDataLoader(self.dataset, self.data_path, True, self.split, task-1, self.batch_size, get_transforms(self.dataset))
                for x, y, _ in prev_data_loader:
                    length += 1
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)
                    if self.ILtype == 'task':
                        y = y % self.increment

                    inj_logit = self.prev_model.forward_inj(self.prev_model.forward_backbone(x))
                    # cross_entropy(inj_logit, y).backward()
                    if prev_avg_K_grad is not None:
                        cross_entropy(inj_logit, y).backward()
                        prev_avg_K_grad += self.prev_model.get_K_grad()
                        prev_avg_bias_grad += self.prev_model.get_bias_grad()
                    else:
                        cross_entropy(inj_logit, y).backward()
                        prev_avg_K_grad = self.prev_model.get_K_grad()
                        prev_avg_bias_grad = self.prev_model.get_bias_grad()
                prev_avg_K_grad /= length
                prev_avg_bias_grad /= length
                K_w_prev = self.prev_model.get_K()
                K_bias_prev = self.prev_model.get_bias()


            '''
            Train one task during configured epoch.
            '''
            # train
            if task == 0:
                train_epoch = 50
            else:
                train_epoch = self.train_epoch
            for epoch in range(train_epoch):
                # Train current Task
                correct, total = 0, 0
                correct_m, total_m = 0, 0
                for batch_idx, (x, y, t) in enumerate(data_loader):
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)
                    if self.ILtype == 'task':
                        y = y % self.increment

                    feature = self.model.forward_backbone(x)

                    if self.ablate_inj:
                        acc_logit = self.model.forward_acc(feature)
                        inj_logit = self.model.forward_acc(feature)
                        L_It = 0
                        L_At = cross_entropy(acc_logit, y)
                    elif self.ablate_acc:
                        inj_logit = self.model.forward_inj(feature)
                        acc_logit = self.model.forward_acc(feature)
                        L_It = cross_entropy(inj_logit, y)
                        L_At = cross_entropy(acc_logit, y)
                    else:
                        inj_logit = self.model.forward_inj(feature)
                        acc_logit = self.model.forward_acc(feature)
                        L_It = cross_entropy(inj_logit, y)
                        L_At = cross_entropy(acc_logit, y)

                    # if task == 0:
                    #     acc_logit = torch.zeros_like(inj_logit).to(self.device)

                    # print(inj_logit)
                    '''
                    L_It and L_At is obtained by the new data.
                    L_It is cross entropy loss value between output of the
                    injection classifier and GT value.
                    L_At is cross entropy loss value between output of the
                    accumulation classifier and GT value.
                    '''
                    # L_It = cross_entropy(inj_logit, y)
                    # L_At = cross_entropy(acc_logit, y)
                    
                    # Train memory if task>0
                    '''
                    The memory is used after first task.
                    At the first task, there are nothing in memory.
                    '''
                    if task > 0:
                        # print(f'prev_K_grad : {prev_avg_K_grad.shape}, K : {self.model.get_K().shape}')
                        # print(f'prev_B_grad : {prev_avg_bias_grad.shape}, B : {self.model.get_bias().shape}')
                        '''
                        L_a value can be calculated 
                        when the previous gradient value exists.
                        This loss can be regarded as the interation with previous task.
                        '''
                        if self.ablate_attn:
                            L_a = 0
                        else:
                            L_a = (torch.abs(torch.tensordot(prev_avg_K_grad, (self.model.get_K() - K_w_prev)))).sum() / 32. + \
                                (torch.abs(torch.tensordot(prev_avg_bias_grad, (self.model.get_bias() - K_bias_prev), dims=([2, 1], [2, 1])))).sum() / 32.
                        
                        '''
                        Calculate the logit value from accumulation classifier on the data in memory buffer.
                        '''                        
                        memory_idx = np.random.permutation(self.memory_size)[:self.batch_size]
                        mx,my,mt,z = self.memory[memory_idx]

                        mx = mx.to(self.device)
                        my = my.type(torch.LongTensor).to(self.device)
                        z = z.to(self.device)

                        if self.ILtype=='task':
                            my = my % self.increment
                            features = self.model.forward_backbone(mx)
                            features_prev = self.prev_model.forward_backbone(mx)
                            L_r = None
                            
                            for i in range(self.batch_size):
                                if L_r is None:
                                    acc_logit = self.model.forward_acc(features[i,...], int(mt[i].item()))
                                    # z = self.prev_model.forward_acc(features_prev[i,...], int(mt[i].item())).unsqueeze(0)
                                    L_r = cross_entropy(acc_logit, my[i,...])
                                    acc_logit = acc_logit.unsqueeze(0)
                                else:
                                    acc_log = self.model.forward_acc(features[i,...], int(mt[i].item()))
                                    # z_ = self.prev_model.forward_acc(features_prev[i,...], int(mt[i].item()))
                                    # z = torch.concat([z, z_.unsqueeze(0)], dim=0)
                                    L_r += cross_entropy(acc_log, my[i,...])
                                    acc_logit = torch.concat([acc_logit, acc_log.unsqueeze(0)], dim=0)
                                    
                            _, predicted_m = torch.max(acc_logit, 1)
                            correct_m += (predicted_m == my).sum().item()
                            total_m += my.size(0)
                            
                        else:
                            acc_logit = self.model.forward_acc(self.model.forward_backbone(mx))
                            z = self.prev_model.forward_acc(self.prev_model.forward_backbone(mx))
                            L_r = cross_entropy(acc_logit, my)
                            _, predicted_m = torch.max(acc_logit, 1)
                            if epoch == 40:
                                print(predicted_m)
                                print(my)
                            correct_m += (predicted_m == my).sum().item()
                            total_m += my.size(0)
                        
                    '''If first task, then only the losses obtained by new data are backpropagated.
                    Or, accumulate the losses from memory such as L_r, L_d into L_l'''
                    if task == 0:
                        if self.ablate_inj:
                            total_loss = L_At
                        elif self.ablate_acc:
                            total_loss = L_It + L_At
                        else:
                            total_loss = L_It + L_At
                    else:
                        # L_r = cross_entropy(acc_logit, my)
                        if self.ILtype == 'class':
                            L_d = kl_divergence(nn.functional.log_softmax((z/self.T), dim=1), self.act(acc_logit[:,:self.cur_classes-self.increment]/self.T))
                        else:
                            L_d = kl_divergence(nn.functional.log_softmax((z/self.T), dim=1), self.act(acc_logit/self.T))
                        if self.ablate_acc:
                            L_l = self.alpha*L_r + self.rt*L_At
                        else:
                            L_l = self.alpha*L_r + self.beta*L_d + self.rt*L_At
                        total_loss = L_l + L_It + self.gamma*L_a
                        
                    # To log the accuracy, calculate that
                    _, predicted = torch.max(inj_logit, 1)
                    correct += (predicted == y).sum().item()
                    total += y.size(0)
                    
                    # print(f'batch {batch_idx} | L_l : {L_l}| L_r : {L_r}| L_d : {L_d}| L_At :{L_At}| L_It : {L_It}| L_a : {L_a}| train_loss :{total_loss}|  accuracy : {100*correct/total}')
                    '''
                    Backward and optimize
                    '''                    
                    self.optimizer.zero_grad()
                    total_loss.backward()
                    # if task == 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                    # else:
                    #     nn.utils.clip_grad_norm_(self.model.parameters(), 10.)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    # print(f'batch : {batch_idx} | L : {total_loss} | L_It : {L_It} | L_d :{L_d} | acc : {acc_logit.max()}')
                '''
                Logging
                '''
                if task == 0:
                    self.logger.info(f'epoch {epoch} | L_At :{L_At:.3f}| L_It : {L_It:.3f}| train_loss :{total_loss:.3f} | accuracy : {100*correct/total:.3f}')
                    print(f'epoch {epoch} | L_At :{L_At:.3f}| L_It : {L_It:.3f}| train_loss :{total_loss:.3f} |  accuracy : {100*correct/total:.3f}')
                else:
                    self.logger.info(f'epoch {epoch} | L_At (acc):{L_At:.3f}| L_It (inj): {L_It:.3f}| L_a (att): {L_a}| L_l (accum): {L_l:.3f}| L_r (replay): {L_r:.3f}| L_d (dark) : {L_d:.3f}|  train_loss :{total_loss:.3f} |  accuracy : {100*correct/total:.3f} | m_accuracy : {100*correct_m/total_m:.3f}')
                    print(f'epoch {epoch} | L_At (acc):{L_At:.3f}| L_It (inj): {L_It:.3f}| L_a (att): {L_a}| L_l (accum): {L_l:.3f}| L_r (replay): {L_r:.3f}| L_d (dark) : {L_d:.3f}|  train_loss :{total_loss:.3f} |  accuracy : {100*correct/total:.3f} | m_accuracy : {100*correct_m/total_m:.3f}')
            

            '''Update memory'''
            conf_score_list = []
            x_list = []
            labels_list = []
            
            '''Calculate confidence score'''
            for x, y, t in data_loader:
                x_list.append(x)
                labels_list.append(y)
                x = x.to(device=self.device)
                y = y.to(device=self.device)
                if self.ILtype == 'task':
                    y = y % self.increment
                feature = self.model.forward_backbone(x)
                if self.ablate_inj:
                    inj_logit = self.model.forward_acc(featrue)
                else:
                    inj_logit = self.model.forward_inj(feature)

                conf_score_list.append(confidence_score(inj_logit.detach(), y.detach()).numpy())
                # store logit z=inj_logit for each x
            
            conf_score = np.array(conf_score_list).flatten()
            labels = torch.cat(labels_list).flatten()
            xs = torch.cat(x_list).view(-1, *x.shape[1:])

            '''To add new examplars, reduce examplars to K'''
            if task > 0:
                self.memory.remove_examplars(K)

            '''Save previous model'''
            self.prev_model = copy.deepcopy(self.model)
            self.prev_model.eval()

            '''Add new examplars'''
            if self.ablate_memupdate:
                conf_score_sorted = np.random.permutation(conf_score)
            else:
                conf_score_sorted = conf_score.argsort()[::-1]
            for label in range(self.increment*task, self.increment*(task+1)):
                new_x = xs[conf_score_sorted[labels==label][:K]]
                new_y = labels[conf_score_sorted[labels==label][:K]]
                new_t = torch.full((K,), task).type(torch.LongTensor)
                new_z = None
                
                for chunk in range(0, new_x.shape[0], self.batch_size):
                    x = new_x[chunk:chunk+self.batch_size]
                    n_samples = x.shape[0]
                    if n_samples < 32:
                        pad_size = self.batch_size - n_samples
                        zero_pad = torch.zeros((pad_size, *x.shape[1:]))
                        x = torch.concat([x, zero_pad])
                    x = x.to(device=self.device)
                    z = self.prev_model.forward_acc(self.prev_model.forward_backbone(x))
                    z = z[:n_samples].detach().cpu()
                    if new_z is None:
                        new_z = z
                    else:
                        new_z = torch.concat([new_z, z], dim=0)
                # print('x shape : ', new_x.shape)
                # print('z shape : ', new_z.shape)
                if self.ILtype == "class":
                    new_z = new_z[:,-self.increment:]
                self.memory.update_memory(label, new_x, new_y, new_t, new_z)
                
            '''updatae r(t)'''
            self.rt *= 0.9

            '''
            After task, the number of output of classifier should be extended.
            In Task IL, LVT generates new classifier
            and store the currently used classifier.
            In Class IL, LVT extendes the classifiers.
            '''
            if self.ILtype == 'task':
                self.model.add_classes(self.increment)
            if self.ILtype == 'class':
                self.model.add_classes(self.increment)
                self.cur_classes += self.increment
            
            
            '''Reset optimizer'''
            self.optimizer = optim.SGD(self.model.parameters(), lr = self.lr)
            if self.scheduler:
                self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, self.train_epoch/10, 0.1)
            
            '''Save model and memory'''
            self.save(self.model, task)
            
            '''test'''
            self.eval(task)
    
    '''
    In this function, just evaluate the model on whole previous tasks 
    where the model is just after trained with current task data.
    '''
    def eval(self, task, test=False):
        self.model.eval()
        acc = []
        with torch.no_grad():
            for task_id in range(task+1):
                correct, total = 0, 0
                data_loader = IncrementalDataLoader(self.dataset, self.data_path, False, self.split, task_id, self.batch_size, get_transforms(self.dataset, True))
                for x, y, t in data_loader:
                    x = x.to(device=self.device)
                    y = y.to(device=self.device)
                    if self.ILtype == 'task':
                        y = y % self.increment
                        acc_logit = self.model.forward_acc(self.model.forward_backbone(x), task_id)
                    else:
                        acc_logit = self.model.forward_acc(self.model.forward_backbone(x))

                    _, predicted = torch.max(acc_logit, 1)
                    # print(predicted)
                    # print(y)
                    correct += (predicted == y).sum().item()
                    total += y.size(0)
                acc.append(100*correct/total)
                self.logger.info(f'Test accuracy on task {task_id} : {100*correct/total}')
                print(toGreen(f'Test accuracy on task {task_id} : {100*correct/total}'))
        self.logger.info(f'Total test accuracy on task {task} : {sum(acc)/len(acc)}')
        print(toGreen(f'Total test accuracy on task {task} : {sum(acc)/len(acc)}'))
        self.model.train()
        if test:
            return acc
        
    
    '''
    Test the model.
    The difference between eval function is that
    this function evaluates the model which is loaded for every task.
    In other words, for each task, the model is loaded according to the task number,
    and it will be evaluated.
    '''
    def test(self):
        self.model.eval()
        result_acc = np.zeros((self.split, self.split))
        with torch.no_grad():
            for task_id in range(self.split):
                '''Load model'''
                self.model = LVT(batch=self.batch_size, n_class=self.increment*(task_id+1), IL_type=self.ILtype, dim=512, num_heads=self.num_head, hidden_dim=self.hidden_dim, bias=self.bias, device=self.device, ablation=self.ablate_attn).to(self.device)
                cur_dir = os.path.dirname(os.path.realpath(__file__))
                model_name = f'{self.ILtype}_{self.dataset}_task_{task_id}.pt'
                self.model = torch.load(os.path.join(os.path.join(cur_dir, self.log_dir, "best_models", model_name)), map_location=self.device)
                self.model.add_classes(self.increment)
                '''evaluation for task task_id'''
                self.logger.info(f'Task {task_id}')
                print(toRed(f'----- Task {task_id} -----'))
                task_result = self.eval(task_id, True)
                result_acc[task_id, :task_id+1] = np.array(task_result)
                
        avg_forgetting = [0]
        for t in range(1, self.split):
            forgetting = []
            for i in range(t):
                forgetting.append(max(result_acc[:,i] - result_acc[t,i]))
            avg_forgetting.append(sum(forgetting)/len(forgetting))
        accuracies = np.sum(result_acc, axis=1)
        accuracies = [accuracies[i-1]/i for i in range(1, self.split+1)]
                
        self.logger.info(f'Result accuracy for each task : {accuracies}')
        print(toGreen(f'Result accuracy for each task : {accuracies}'))
        self.logger.info(f'Forgetting for each task : {avg_forgetting}')
        print(toGreen(f'Forgetting for each task : {avg_forgetting}'))