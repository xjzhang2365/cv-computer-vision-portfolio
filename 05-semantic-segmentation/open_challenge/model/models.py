import torch
from torch import nn
import torch.nn.functional as F
from .resnet import *
from .mobilenetv3 import *


class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)


class MobileNetV3(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV3, self).__init__()
        from functools import partial
        
        model = mobilenetv3_large(pretrained)
       # model = mobilenetv3_small(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        #self.down_idx = [2, 3, 4, 5,6,7,8]
        self.down_idx = [2，4，7，14]
        if downsample_factor == 8:
            for i in range(self.down_idx[-2], self.down_idx[-1]):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=4))
        elif downsample_factor == 16:
            for i in range(self.down_idx[-1], self.total_idx):
                self.features[i].apply(partial(self._nostride_dilate, dilate=2))

        
    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m.stride == (2, 2):
                m.stride = (1, 1)
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate//2, dilate//2)
                    m.padding = (dilate//2, dilate//2)
            else:
                if m.kernel_size == (3, 3):
                    m.dilation = (dilate, dilate)
                    m.padding = (dilate, dilate)
    def forward(self, x):
            # print("In MobileNetV3, the input's shape is: ", x.size())
        # x_aux = self.features(x)
        # # print("In MobileNetV3, the x_aux's shape is: ", x_aux.size())
        # x = self.features(x)
        # x_aux = self.features[:14](x)
        # x = self.features[14:](x_aux)
        x_aux = self.features[:14](x)
        x = self.features[14:](x_aux)
        return x_aux, x

class PSPNet(nn.Module):
    def __init__(self, layers=101, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True, args=None):
        super(PSPNet, self).__init__()
        assert layers in [18, 50, 101, 152]
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = args.use_ppm
        self.use_aspp = args.use_aspp
        self.criterion = criterion
        self.args = args
        self.layers_label = layers


        
        if self.layers_label == 'mv3':
            ## If backbone is MobileNetV2, code below is used
            downsample_factor = 8
            fea_dim = 160
            self.backbone = MobileNetV3(downsample_factor, pretrained)
        elif layers == 101:
            resnet = resnet101(pretrained=pretrained)
            fea_dim = 2048
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
            for n, m in self.layer3.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                        m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv2' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)

        if self.use_ppm: 
            self.enrich_module = PPM(fea_dim, int(fea_dim/len(bins)), bins)
        
        fea_dim *= 2

   

        if self.layers_label == 'mv3':
            self.cls = nn.Sequential(
                nn.Conv2d(fea_dim, 80, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(80),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(80, classes, kernel_size=1)
            )
        else:               
            self.cls = nn.Sequential(
                nn.Conv2d(fea_dim, 512, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(512, classes, kernel_size=1)
                )

        if self.layers_label == 'mv3':
            self.aux = nn.Sequential(
                nn.Conv2d(160, 40, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(40),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(40, classes, kernel_size=1)
            )
        else:
            self.aux = nn.Sequential(
                # nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
                nn.Conv2d(1024, 256, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Dropout2d(p=dropout),
                nn.Conv2d(256, classes, kernel_size=1)
            )

    def forward(self, x, y=None, preact=False):

        ### [requires implementation] forward and get predicted logits

        x_size = x.size()
        assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        h = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        w = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)

        if self.layers_label == 'mv3':
            ## If backbone is mobilenet, code below is used
            aux, x = self.backbone(x)
        else:
            # If backbone is ResNet, code below is used
            x = self.layer0(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x_tmp = self.layer3(x)
            x = self.layer4(x_tmp)

        if self.use_ppm: 
            x = self.enrich_module(x)
        
        x = self.cls(x)
        
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            if self.layers_label == 'mv3':
                aux = self.aux(aux)
            else:
                aux = self.aux(x_tmp)
            if self.zoom_factor != 1:
                aux = F.interpolate(aux, size=(h, w), mode='bilinear', align_corners=True)
                main_loss = self.criterion(x, y)
                aux_loss = self.criterion(aux, y)
            return x.max(1)[1], main_loss, aux_loss
        else:
            return x
        