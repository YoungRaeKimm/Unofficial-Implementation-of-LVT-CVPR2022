import torch
import torch.nn.functional as F 
from torch import nn 
from torchvision import models
import copy

from einops import rearrange
from models.archs.classifiers import Classifier

class Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        # self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.to_qv = nn.Linear(dim, dim * 2, bias = False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        # self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        # Learnable External Key
        self.k = nn.Linear(dim, dim, bias = False)
        # Learnable Attention Bias
        self.bias = nn.Parameter(torch.rand(dim))
        

    def forward(self, x):
        b,c,h,w = x.shape

        qv = self.to_qv(x)
        q,v = qv.chunk(2, dim=1)
        print(q.size())

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(self.k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        bias = rearrange(self.bias, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # q = torch.nn.functional.normalize(q, dim=-1)
        # k = torch.nn.functional.normalize(k, dim=-1)
        v = torch.nn.functional.batch_norm(v, dim=-1)
        # print("q shape")
        # print(q.shape)
        # print("k shape")
        # print(k.shape)
        
        attn = (q @ k.transpose(-2, -1) + bias) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, hidden_dim, bias):
        super(TransformerBlock, self).__init__()

        self.attn = Attention(dim, num_heads, bias)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.ffn = FeedForward(dim, hidden_dim, bias)

    def forward(self, input):
        out = F.batch_norm(self.conv(self.attn(input)),  dim=-1)
        out += input
        out += self.ffn(out) + out

        return out
    
class Backbone(nn.Module):
    def __init__(self):
        super(Backbone, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(list(self.backbone.modules())[0])
        
    def forward(self, x):
        return self.backbone(x)
    
class LVT(nn.Module):
    def __init__(self, n_class, IL_type, dim, num_heads, hidden_dim, bias):
        self.IL_type = IL_type
        self.dim = dim
        self.backbone = Backbone()
        self.stage1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=num_heads, hidden_dim=hidden_dim, bias=bias) for i in range(2)])
        self.shrink1 = nn.Conv2d(dim, dim*2, kernel_size=3, stride=2, padding=1, bias=bias)
        self.stage2 = nn.Sequential(*[TransformerBlock(dim=dim*2, num_heads=num_heads, hidden_dim=hidden_dim, bias=bias) for i in range(2)])
        self.shrink2 = nn.Conv2d(dim*2, dim*4, kernel_size=3, stride=2, padding=1, bias=bias)
        self.stage3 = nn.Sequential(*[TransformerBlock(dim=dim*4, num_heads=num_heads, hidden_dim=hidden_dim, bias=bias) for i in range(2)])
        self.injection_classifier = Classifier(features_dim = dim*4, n_classes = n_class, init = 'kaiming')
        self.accumulation_classifier = Classifier(features_dim = dim*4, n_classes = n_class, init = 'kaiming')
        self.K_w = torch.concat([
            self.stage1[0].attn.k.weight,
            self.stage1[1].attn.k.weight,
            self.stage2[0].attn.k.weight,
            self.stage2[1].attn.k.weight,
            self.stage3[0].attn.k.weight,
            self.stage3[1].attn.k.weight
        ])
        self.B = torch.concat([
            self.stage1[0].attn.bias,
            self.stage1[1].attn.bias,
            self.stage2[0].attn.bias,
            self.stage2[1].attn.bias,
            self.stage3[0].attn.bias,
            self.stage3[1].attn.bias
        ])
        
        if self.IL_type == 'task':
            self.prev_acc_classifiers = []
        
    
    def get_K(self):
        return torch.concat([
            self.stage1[0].attn.k.weight,
            self.stage1[1].attn.k.weight,
            self.stage2[0].attn.k.weight,
            self.stage2[1].attn.k.weight,
            self.stage3[0].attn.k.weight,
            self.stage3[1].attn.k.weight
        ])
    
    def get_bias(self):
        return torch.concat([
            self.stage1[0].attn.bias,
            self.stage1[1].attn.bias,
            self.stage2[0].attn.bias,
            self.stage2[1].attn.bias,
            self.stage3[0].attn.bias,
            self.stage3[1].attn.bias
        ])
        
    def get_K_grad(self):
        return torch.concat([
            self.stage1[0].attn.k.weight.grad,
            self.stage1[1].attn.k.weight.grad,
            self.stage2[0].attn.k.weight.grad,
            self.stage2[1].attn.k.weight.grad,
            self.stage3[0].attn.k.weight.grad,
            self.stage3[1].attn.k.weight.grad
        ])
    
    def get_bias_grad(self):
        return torch.concat([
            self.stage1[0].attn.bias.grad,
            self.stage1[1].attn.bias.grad,
            self.stage2[0].attn.bias.grad,
            self.stage2[1].attn.bias.grad,
            self.stage3[0].attn.bias.grad,
            self.stage3[1].attn.bias.grad
        ])
            
    def forward_backbone(self, input):
        out = self.backbone(input)
        out = self.shrink1(self.stage1(out))
        out = self.shrink2(self.stage2(out))
        out = F.adaptive_avg_pool2d(out, (1, 1))
        return out
        
    def forward_inj(self, input, task_id=None):
        return self.injection_classifier(input)
        
    def forward_acc(self, input, task_id=None):
        if task_id is None:
            return self.accumulation_classifier(input)
        else:
            return self.prev_acc_classifiers[task_id](input)
        
    def add_classes(self, n_class):
        # self.injection_classifier.add_classes(n_class)
        self.injection_classifier = Classifier(features_dim = self.dim*4, n_classes = n_class, init = 'kaiming')
        if self.IL_type == 'task':
            self.prev_acc_classifiers.append(copy.deepcopy(self.accumulation_classifier))
        else:
            self.accumulation_classifier.add_classes(n_class)