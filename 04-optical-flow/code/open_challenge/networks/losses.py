import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class OursLoss(nn.Module):
    def __init__(self, args, div_flow = 0.05):
        super(OursLoss, self).__init__()
        self.div_flow = div_flow 
        self.loss_labels = ['Ours'],

        self.numScales = 3
        
        self.loss_weights = torch.FloatTensor([(0.16 /2 ** scale) for scale in range(self.numScales)])

    def forward(self, output, target):
        lossvalue = 0
        epevalue = 0
        target = self.div_flow * target
        for i, output_ in enumerate(output):
            target_ = F.interpolate(target, output_.shape[2:], mode='bilinear', align_corners=False)
            assert output_.shape == target_.shape, (output_.shape, target_.shape)
            ''' Implement the MultiScale loss here'''
            lossvalue = torch.norm(output_-target_, p=2, dim=1).mean()

            epevalue += self.loss_weights[i] * lossvalue

            ''''''
        return [epevalue]
