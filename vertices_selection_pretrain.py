import numpy as np
from skimage.exposure import match_histograms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, resnet34

class ModifiedResNet34(nn.Module):
    def __init__(self, in_channels=3, strides=[1, 2, 2, 2], pretrained=True):
        super(ModifiedResNet34, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet = resnet34(pretrained=pretrained)
        if in_channels == 3:
            self.conv1.weight = resnet.conv1.weight
        else:
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1, keepdim=True))
        self.bn1 = resnet.bn1
        self.relu = resnet.relu 
        self.maxpool = resnet.maxpool
        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=strides[0])
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=strides[1])
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=strides[2])
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=strides[3])
        self._load_pretrained_weights(resnet)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _load_pretrained_weights(self, resnet):
        own_state = self.state_dict()
        for name, param in resnet.state_dict().items():
            if name not in own_state or 'fc' in name:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                pass

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

class BlueLayerResNet(nn.Module):
    def __init__(self):
        super(BlueLayerResNet, self).__init__()
        self.model = ModifiedResNet34(in_channels=1, strides=[1, 2, 2, 2], pretrained=True)
    def forward(self, x):
        return self.model(x)

class RGBLayerResNet(nn.Module):
    def __init__(self):
        super(RGBLayerResNet, self).__init__()
        self.model = ModifiedResNet34(in_channels=3, strides=[1, 1, 2, 2], pretrained=True)

    def forward(self, x):
        return self.model(x)

class MixLayerResNet(nn.Module):
    def __init__(self):
        super(MixLayerResNet, self).__init__()
        self.model = ModifiedResNet34(in_channels=1, strides=[1, 1, 2, 1], pretrained=True)

    def forward(self, x):
        return self.model(x)


class MultiScaleSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(MultiScaleSelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, channels, height, width = x.size()

        query = self.query_conv(x).view(batch_size, -1, height * width)  
        key = self.key_conv(x).view(batch_size, -1, height * width)  
        value = self.value_conv(x).view(batch_size, -1, height * width)  

        attention = F.softmax(torch.bmm(query.permute(0, 2, 1), key), dim=-1)  

        out = torch.bmm(value, attention.permute(0, 2, 1))  
        out = out.view(batch_size, channels, height, width)  

        return self.gamma * out + x  

class VerticesSelection(nn.Module):
    def __init__(self, device):
        super(VerticesSelection, self).__init__()
        self.BlueLayerResNet = BlueLayerResNet()
        self.RGBLayerResNet = RGBLayerResNet()
        self.MixLayerResNet = MixLayerResNet()
        self.attention16 = MultiScaleSelfAttention(in_channels=512) 
        self.attention32 = MultiScaleSelfAttention(in_channels=512) 
        self.attention64 = MultiScaleSelfAttention(in_channels=512)

        self.to(device)

    def process_channels(self, images):
        red_channel = images[:, 0, :, :].unsqueeze(1).cpu().numpy()
        green_channel = images[:, 1, :, :].unsqueeze(1).cpu().numpy()
        blue_channel = images[:, 2, :, :].unsqueeze(1)
        matched_batch = np.zeros_like(red_channel)
        for i in range(red_channel.shape[0]):
            matched_batch[i] = match_histograms(red_channel[i], green_channel[i])
        mix_channel = torch.from_numpy(matched_batch).float().to(images.device)
        return mix_channel, blue_channel.to(images.device)

    def forward(self, images):
        mix_channel, blue_channel = self.process_channels(images)
        rgb_channel = images
        v1 = self.BlueLayerResNet(blue_channel)
        v2 = self.RGBLayerResNet(rgb_channel)
        v3 = self.MixLayerResNet(mix_channel)
        v1 = self.attention16(v1)
        v2 = self.attention32(v2)
        v3 = self.attention64(v3)
        v1 = v1.permute(0, 2, 3, 1)  
        v2 = v2.permute(0, 2, 3, 1)
        v3 = v3.permute(0, 2, 3, 1)

        return v1, v2, v3
