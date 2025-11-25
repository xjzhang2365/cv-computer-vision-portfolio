import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

class Compose(object):
    def __init__(self, co_transforms):
        self.co_transforms = co_transforms
    def __call__(self, input, target):
        for t in self.co_transforms:
            input, target = t(input,target)

        return input, target