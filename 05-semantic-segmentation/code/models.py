import torch
from torch import nn
import torch.nn.functional as F
from .resnet import *
from .mobilenet import *

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

class ASPP(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(ASPP, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.Conv2d(in_dim, reduction_dim, kernel_size=3, bias=False, padding=bin, dilation=bin),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            # out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
            out.append(f(x))
        return torch.cat(out, 1)


class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            nn.BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )


class SpatialGather_Module(nn.Module):
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c 
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        ocr_context = torch.matmul(probs, feats)\
        .permute(0, 2, 1).unsqueeze(3)# batch x k x c
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)   

        # add bg context
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 scale=1, 
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale, 
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    def __init__(self, 
                 in_channels, 
                 key_channels, 
                 out_channels, 
                 scale=1, 
                 dropout=0.1, 
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels, 
                                                           key_channels, 
                                                           scale, 
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output

class OCR(nn.Module):
    def __init__(self, in_dim, ocr_mid_channels, ocr_key_channels, classes):
        super(OCR, self).__init__()

        self.aux_head = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim, classes, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(in_dim, ocr_mid_channels,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ocr_mid_channels),
            nn.ReLU(inplace=True),
        )
        self.ocr_gather_head = SpatialGather_Module(classes)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels,
                                                 key_channels=ocr_key_channels,
                                                 out_channels=ocr_mid_channels,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )

    def forward(self, x):
        x_size = x.size()
        out_aux = self.aux_head(x)
        feats = self.conv3x3_ocr(x)
        context = self.ocr_gather_head(feats, out_aux)
        feats =  self.ocr_distri_head(feats, context)
        return feats



## MobileNetV2
class MobileNetV2(nn.Module):
    def __init__(self, downsample_factor=8, pretrained=True):
        super(MobileNetV2, self).__init__()
        from functools import partial
        
        model = mobilenetv2(pretrained)
        self.features = model.features[:-1]

        self.total_idx = len(self.features)
        self.down_idx = [2, 4, 7, 14]

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
        x_aux = self.features[:14](x)
        x = self.features[14:](x_aux)
        return x_aux, x


class PSPNet(nn.Module):
    def __init__(self, layers=18, bins=(1, 2, 3, 6), dropout=0.1, classes=2, zoom_factor=8, use_ppm=True, criterion=nn.CrossEntropyLoss(ignore_index=255), pretrained=True, args=None):
        super(PSPNet, self).__init__()
        #assert layers in [18, 50, 101, 152]# comment this line when applying mobilenet
        assert 2048 % len(bins) == 0
        assert classes > 1
        assert zoom_factor in [1, 2, 4, 8]
        self.zoom_factor = zoom_factor
        self.use_ppm = args.use_ppm
        self.use_aspp = args.use_aspp
        self.use_ocr = args.use_ocr
        self.criterion = criterion
        self.args = args
        self.layers_label = layers

        ### [requires implementation] parameter/module definitions
        '''
        You are required to reproduce PPM of PSPNet. Also, the you should also implement the ASPP module used in DeepLab-V3+
        PSPNet: https://jiaya.me/papers/PSPNet_cvpr17.pdf
        DeepLab-V3+: https://arxiv.org/abs/1802.02611
        If you correctly implement both PPM and ASPP module, they should achieve close performance (gap <= 2%). 
        
        Dilations should be added to the model to make sure the output feature map is roughly 1/8 of the input feature.
            Example: if the input spatial size is 473x473 and the output feature map's spatial size should be 60x60.
        
        For resnet-based backbones, you can reuse the pretrained layer like:
        self.layer0 = nn.Sequential(network.conv1, network.bn1, network.relu, network.conv2, network.bn2, network.relu, network.conv3, network.bn3, network.relu, network.maxpool)

        For mobilenet-based backbone, you can reuse the pretrained layer like:
        self.layer0 = nn.Sequential(*[network.features[_i] for _i in [xxx]])
        '''


        # If backbone is ResNet, code below is used
        if self.layers_label == 18:
            resnet = resnet18(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        if self.layers_label == 50:
            resnet = resnet50(pretrained=pretrained)
            self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.conv2, resnet.bn2, resnet.relu, resnet.conv3, resnet.bn3, resnet.relu, resnet.maxpool)
            self.layer1, self.layer2, self.layer3, self.layer4 = resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        if self.layers_label == 'mv2':
            ## If backbone is MobileNetV2, code below is used
            downsample_factor = 8
            fea_dim = 320
            self.backbone = MobileNetV2(downsample_factor, pretrained)

        if self.layers_label == 18:
            fea_dim = 512
            for n, m in self.layer3.named_modules():
                if 'conv1' in n:
                    m.dilation, m.padding, m.stride = (2, 2), (2, 2), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
            for n, m in self.layer4.named_modules():
                if 'conv1' in n:
                    m.dilation, m.padding, m.stride = (4, 4), (4, 4), (1, 1)
                elif 'downsample.0' in n:
                    m.stride = (1, 1)
        elif self.layers_label == 50:
            fea_dim = 2048
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
        elif self.use_aspp:
            self.enrich_module = ASPP(fea_dim, int(fea_dim/len(bins)), bins)
        elif self.use_ocr:
            ocr_mid_channels = 256
            ocr_key_channels = 512
            self.enrich_module = OCR(fea_dim, ocr_mid_channels, ocr_key_channels, classes)
        
        fea_dim *= 2

        if self.use_ocr:
            self.cls = nn.Conv2d(256, classes, 
                kernel_size=1, stride=1, padding=0, bias=True)

        else:
            if self.layers_label == 'mv2':
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

        if self.layers_label == 'mv2':
            self.aux = nn.Sequential(
                nn.Conv2d(96, 40, kernel_size=3, padding=1, bias=False),
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

        if self.layers_label == 'mv2':
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
        elif self.use_aspp:
            x = self.enrich_module(x)
        elif self.use_ocr:
            x = self.enrich_module(x)
        
        x = self.cls(x)
        
        if self.zoom_factor != 1:
            x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)

        if self.training:
            if self.layers_label == 'mv2':
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