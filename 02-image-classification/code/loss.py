import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    # TODO: task 2
    def __init__(self, **kwargs):
        super(CrossEntropyLoss, self).__init__()
        # TODO: implemente cross entropy loss for task2;
        # You cannot directly use any loss functions from torch.nn or torch.nn.functional, other modules are free to use.

    def forward(self, x, y, **kwargs):
        
        
        loss = -1.*x.gather(1, y.unsqueeze(-1)) + torch.log(torch.exp(x).sum(dim=1))
        

        return torch.mean(loss)
        
        
        
