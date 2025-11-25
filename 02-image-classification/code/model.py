import torch
import torch.nn as nn
import torch.nn.functional as F
configures = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class vggnet(nn.Module):
# TODO: task 1
    
    def __init__(self,cfg, num_classes):

        super(vggnet,self).__init__()
        
        self.conv_layers = self.make_conv_layers(configures[cfg])
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7, 1300),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1300,1300),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1300,num_classes)
            )
           

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.classifier(x)
        return x

    def make_conv_layers(self, architecture):
        layers = []
        in_channels = 3

        for v in architecture:
            if type(v) == int:
                out_channels = v
                layers += [nn.Conv2d(in_channels=in_channels,out_channels=out_channels,
                                    kernel_size=3, stride=1, padding=1),
                           nn.BatchNorm2d(v),
                           nn.ReLU(inplace=True)]
                in_channels = v
            elif v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2,stride=2)]

        return nn.Sequential(*layers)

    
