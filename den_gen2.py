import torch
import torch.nn as nn
from torchvision.models import resnet152


class AuxConv(nn.Module):
    def __init__(self, in_channels, c_tag, stride=1, p=0, downsample=False):
        super(AuxConv, self).__init__()
        self.aux = nn.Sequential(nn.Conv2d(in_channels, c_tag, kernel_size=(3, 1)),
                                 nn.ReLU(),
                                 nn.Dropout(p),
                                 nn.Conv2d(c_tag, c_tag, kernel_size=(1, 3)),
                                 nn.ReLU(),
                                 nn.Dropout(p))
        if downsample:
            self.aux.add_module('downsample',
                nn.Conv2d(c_tag, c_tag, kernel_size=3, stride=2))

    def forward(self, input):
        return self.aux(input)


class DEN(nn.Module):
    def __init__(self, backbone_wts=None, backbone_freeze=True, p=0):
        super(DEN, self).__init__()

        resnet = resnet152(pretrained=False)
        if backbone_wts != None:
            resnet = self._init_resnet(resnet, backbone_wts)
        
        if backbone_freeze:
            for param in resnet.parameters():
                param.requires_grad = False
            
        
        # prepare the network
        self._flat_resnet152(resnet)

        aux_1024 = [AuxConv(in_channels=1024, c_tag=16, p=p, downsample=True) for _ in range(13)]
        aux_2048 = [AuxConv(in_channels=2048, c_tag=16, p=p) for _ in range(3)]
        self.aux_modules = nn.ModuleList(aux_1024 + aux_2048)
        
        self._init_added_weights()
        
    def _init_resnet(self, resnet, backbone_wts):
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, 25 * 32)
        resnet.load_state_dict(torch.load(backbone_wts))

        return resnet


    def _init_added_weights(self):
        for name,param in self.aux_modules.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
    
    
    def _flat_resnet152(self, model):
        
        # break the resent to its building blocks
        # into a list
        flattened = []
        flattened += list(model.children())[:4]

        for i in range(4,8):
            sequence = list(model.children())[i]
            flattened += list(sequence.children())

        flattened += list(model.children())[-2:]

        self.resnet_top = nn.Sequential(*flattened[:38])
        self.resnet_mid = nn.ModuleList(flattened[38:54])
        self.avg_pool2d = flattened[54]
        self.deconv = nn.Sequential(
                            self._deconv_block(in_channels=256, kernel_size=3, stride=2, padding=1),
                            self._deconv_block(in_channels=64, kernel_size=3, stride=2, padding=[2,1]),
                            self._deconv_block(in_channels=16, kernel_size=3, stride=2, padding=[2,1]),
                            self._deconv_block(in_channels=4, kernel_size=[3,4], stride=1, padding=2))
        
        
    def _deconv_block(self, in_channels, kernel_size, stride, padding):
        return nn.Sequential(
                    nn.ConvTranspose2d(in_channels, in_channels,kernel_size,
                                       stride, padding),
                    nn.BatchNorm2d(in_channels),
                    nn.ReLU(),
                    nn.Conv2d(in_channels, in_channels//2, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(in_channels//2),
                    nn.ReLU(),
                    nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(in_channels//2),
                    nn.ReLU(),
                    nn.Conv2d(in_channels//2, in_channels//2, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(in_channels//2),
                    nn.ReLU(),
                    nn.Conv2d(in_channels//2, in_channels//4, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(in_channels//4),
                    nn.ReLU()
                    )

    
    def forward(self, input):
        
        x = self.resnet_top(input)
        
        outputs = []
        for i, block in enumerate(self.resnet_mid):
            x = block(x)
            outputs.append(self.aux_modules[i](x))

        x = torch.cat(outputs, dim=1)
        x = self.deconv(x)
        x = x.view(x.shape[0], -1)

        return x
