import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

class FlowNetEncoder(nn.Module):
    def __init__(self, args, input_channels = 6, div_flow=20):
        super(FlowNetEncoder,self).__init__()

        self.rgb_max = args.rgb_max
        self.div_flow = div_flow      # A coefficient to obtain small output value for easy training, ignore it
        '''Implement Codes here'''
        
        self.conv1 = conv(input_channels, 64, kernel_size=7, stride=2)
        self.conv2 = conv(64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(128, 256, kernel_size=5, stride=2)
        self.conv4 = conv(256, 512, stride=2)
        self.conv4_1 = conv(512, 512)
         
        self.predict_flow4 = predict_flow(512)
        
        ''''''

    def forward(self, inputs):
        ## input normalization
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)
        ##
        '''Implement Codes here'''
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))

        predict_flow4 = self.predict_flow4(out_conv4)
        flow4 = nn.functional.interpolate(predict_flow4, scale_factor=16)


        ''''''

        if self.training:
            return flow4
        else:
            return flow4 * self.div_flow

def conv(input_channels, output_channels, kernel_size=3, stride=1):
    return nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.1, inplace=True)
        )
 
def predict_flow(input_channels):

    return nn.Conv2d(input_channels, 2, kernel_size=3, stride=1, padding=1, bias=True)



