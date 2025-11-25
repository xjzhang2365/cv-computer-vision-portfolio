import torch
import torch.nn as nn
from torch.nn import init

import math
import numpy as np

from .correlation_package.correlation import Correlation

class FlowNetOurs(nn.Module):
    def __init__(self, args, input_channels = 6, div_flow=20, batchNorm=True):
        super(FlowNetOurs, self).__init__()

        self.rgb_max = args.rgb_max
        self.div_flow = div_flow    # A coefficient to obtain small output value for easy training, ignore it

        self.batchNorm = batchNorm
        self.conv1 = conv(input_channels, 64, kernel_size=7, stride=2)
        self.conv2 = conv(64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(128, 256, kernel_size=5, stride=2)
        self.conv4 = conv(256, 512, stride=2)
        self.conv4_1 = conv(512, 512)

        self.predict_flow4 = predict_flow(512)
        self.predict_flow3 = predict_flow(514)
        self.predict_flow2 = predict_flow(258) 

        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2,2,4,2,1,bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2,2,4,2,1,bias=False)

        self.deconv3 = deconv(512,256)
        self.deconv2 = deconv(514,128)

        '''METHOD2-flownetSD
        self.conv1 = conv(input_channels, 64, kernel_size=7, stride=2)
        self.conv2 = conv(64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(128, 256, kernel_size=5, stride=2)
        self.conv4 = conv(256, 512, stride=2)
        self.conv4_1 = conv(512, 512)

        self.predict_flow4 = predict_flow(512)
        self.predict_flow3 = predict_flow(256)
        self.predict_flow2 = predict_flow(128) 

        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2,2,4,2,1,bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2,2,4,2,1,bias=False)

        self.deconv3 = deconv(512,256)
        self.deconv2 = deconv(514,128)

        self.inter_conv3 = i_conv(514,256)
        self.inter_conv2 = i_conv(258,128)
        '''
        
        '''METHOD3-correlation
        self.batchNorm = batchNorm
        self.conv1 = conv(3, 64, kernel_size=7, stride=2)
        self.conv2 = conv(64, 128, kernel_size=5, stride=2)
        self.conv3 = conv(128, 256, kernel_size=5, stride=2)
        self.conv_redir = conv(256, 32, kernel_size=1, stride=1)

        if args.fp16:
            self.corr = nn.Sequential(
                tofp32(),
                Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1),
                tofp16())
        else:
            self.corr = Correlation(pad_size=20, kernel_size=1, max_displacement=20, stride1=1, stride2=2, corr_multiply=1)

        self.corr_activation = nn.LeakyReLU(0.1, inplace=True)
        self.conv4 = conv(473, 512, stride=2)
        self.conv4_1 = conv(512, 512)

        self.predict_flow4 = predict_flow(512)
        self.predict_flow3 = predict_flow(731)
        self.predict_flow2 = predict_flow(258) 

        self.upsampled_flow4_to_3 = nn.ConvTranspose2d(2,2,4,2,1,bias=False)
        self.upsampled_flow3_to_2 = nn.ConvTranspose2d(2,2,4,2,1,bias=False)

        self.deconv3 = deconv(512,256)
        self.deconv2 = deconv(731,128)
        '''



    def forward(self, inputs):
        ## input normalization
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        
        x = torch.cat( (x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)

        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))

        flow4 = self.predict_flow4(out_conv4)
        flow4_to_3 = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(out_conv4)

        cancat3 = torch.cat((out_conv3,out_deconv3,flow4_to_3),1)
        flow3 = self.predict_flow3(cancat3)
        flow3_to_2 = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(cancat3)

        cancat2 = torch.cat((out_conv2,out_deconv2,flow3_to_2),1)
        predict_flow2 = self.predict_flow2(cancat2)


        flow2 = nn.functional.interpolate(predict_flow2,scale_factor=4, mode='bicubic')

        '''METHOD2-flownetSD
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))


        flow4 = self.predict_flow4(out_conv4)
        flow4_to_3 = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(out_conv4)

        concat3 = torch.cat((out_conv3,out_deconv3,flow4_to_3),1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3 = self.predict_flow3(out_interconv3)
        flow3_to_2 = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_to_2),1)
        out_interconv2 = self.inter_conv2(concat2) 
        predict_flow2 = self.predict_flow2(out_interconv2)  
        flow2 = nn.functional.interpolate(predict_flow2,scale_factor=4)     
        '''
        
        
        '''METHOD3-correlation
        x1 = x[:,:,0,:,:]
        x2 = x[:,:,1,:,:]

        out_conv1a = self.conv1(x1)
        out_conv2a = self.conv2(out_conv1a)
        out_conv3a = self.conv3(out_conv2a)
        out_conv1b = self.conv1(x2)
        out_conv2b = self.conv2(out_conv1b)
        out_conv3b = self.conv3(out_conv2b)

        # Merge streams
        out_corr = self.corr(out_conv3a, out_conv3b)
        out_corr = self.corr_activation(out_corr)
        out_conv_redir = self.conv_redir(out_conv3a)

        out_conv3 = torch.cat((out_conv_redir, out_corr), 1)
        out_conv4 = self.conv4_1(self.conv4(out_conv3))


        flow4 = self.predict_flow4(out_conv4)
        flow4_to_3 = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(out_conv4)

        cancat3 = torch.cat((out_conv3,out_deconv3,flow4_to_3),1)
        flow3 = self.predict_flow3(cancat3)
        flow3_to_2 = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(cancat3)

        cancat2 = torch.cat((out_conv2a,out_deconv2,flow3_to_2),1)
        predict_flow2 = self.predict_flow2(cancat2)

        flow2 = nn.functional.interpolate(predict_flow2,scale_factor=4)
        '''

        if self.training:
            return flow2, flow3, flow4
        else:
            return flow2 * self.div_flow

def conv(input_channels, output_channels, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
        )
def conv_bn(batchNorm, input_channels, output_channels, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(input_channels, output_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
        nn.BatchNorm2d(output_channels),
        nn.LeakyReLU(0.1, inplace=True)
        )

def predict_flow(input_channels):

    return nn.Conv2d(input_channels, 2, kernel_size=3, stride=1, padding=1, bias=True)

def deconv(input_channels, output_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(input_channels, output_channels, kernel_size=4, stride=2, padding=1, bias=True),
        nn.LeakyReLU(0.1, inplace=True)
        )

def i_conv(in_planes, out_planes, kernel_size=3, stride=1, bias = True):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=bias),
        )