import numpy as np
from skimage.exposure import match_histograms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, BasicBlock, resnet34

# 定义一个可以修改步长的 ResNet-34
class ModifiedResNet34(nn.Module):
    def __init__(self, in_channels=3, strides=[1, 2, 2, 2], pretrained=True):
        super(ModifiedResNet34, self).__init__()
        self.inplanes = 64
        # 修改第一层卷积层
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 加载预训练模型
        resnet = resnet34(pretrained=pretrained)
        if in_channels == 3:
            self.conv1.weight = resnet.conv1.weight
        else:
            # 调整权重
            with torch.no_grad():
                self.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1, keepdim=True))
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        # 根据 strides 参数创建各层
        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=strides[0])
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=strides[1])
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=strides[2])
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=strides[3])

        # 加载预训练权重到自定义的层中
        self._load_pretrained_weights(resnet)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        # 判断是否需要 downsample
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        # 第一个 block，可能需要 downsample
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # 剩余的 blocks
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _load_pretrained_weights(self, resnet):
        # 将预训练权重复制到新模型中
        own_state = self.state_dict()
        for name, param in resnet.state_dict().items():
            if name not in own_state or 'fc' in name:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                # 忽略形状不匹配的参数
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

# 修改后的 BlueLayerResNet
class BlueLayerResNet(nn.Module):
    def __init__(self):
        super(BlueLayerResNet, self).__init__()
        # 使用默认的 strides，与原始 ResNet-34 一致
        self.model = ModifiedResNet34(in_channels=1, strides=[1, 2, 2, 2], pretrained=True)

    def forward(self, x):
        return self.model(x)

# 修改后的 RGBLayerResNet
class RGBLayerResNet(nn.Module):
    def __init__(self):
        super(RGBLayerResNet, self).__init__()
        # 修改 layer2 的 stride 为 1
        self.model = ModifiedResNet34(in_channels=3, strides=[1, 1, 2, 2], pretrained=True)

    def forward(self, x):
        return self.model(x)

# 修改后的 MixLayerResNet
class MixLayerResNet(nn.Module):
    def __init__(self):
        super(MixLayerResNet, self).__init__()
        # 修改 layer2 和 layer4 的 stride 为 1
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

        # 生成 Q, K, V
        query = self.query_conv(x).view(batch_size, -1, height * width)  # (B, C/8, H*W)
        key = self.key_conv(x).view(batch_size, -1, height * width)  # (B, C/8, H*W)
        value = self.value_conv(x).view(batch_size, -1, height * width)  # (B, C, H*W)

        # 计算注意力权重
        attention = F.softmax(torch.bmm(query.permute(0, 2, 1), key), dim=-1)  # (B, H*W, H*W)

        # 生成增强的特征图
        out = torch.bmm(value, attention.permute(0, 2, 1))  # (B, C, H*W)
        out = out.view(batch_size, channels, height, width)  # (B, C, H, W)

        return self.gamma * out + x  # 残差连接

# 定义 VerticesSelection
class VerticesSelection(nn.Module):
    def __init__(self, device):
        super(VerticesSelection, self).__init__()
        self.BlueLayerResNet = BlueLayerResNet()
        self.RGBLayerResNet = RGBLayerResNet()
        self.MixLayerResNet = MixLayerResNet()
        self.attention16 = MultiScaleSelfAttention(in_channels=512)  # 对应[4, 512, 16, 16]
        self.attention32 = MultiScaleSelfAttention(in_channels=512)  # 对应[4, 512, 32, 32]
        self.attention64 = MultiScaleSelfAttention(in_channels=512)  # 对应[4, 512, 64, 64]

        self.to(device)

    # 处理通道
    def process_channels(self, images):
        # images 形状为 [batch_size, 3, H, W]
        # 将红色和绿色通道移动到 CPU 并转换为 numpy 数组
        red_channel = images[:, 0, :, :].unsqueeze(1).cpu().numpy()
        green_channel = images[:, 1, :, :].unsqueeze(1).cpu().numpy()
        # 蓝色通道保持在 GPU 上
        blue_channel = images[:, 2, :, :].unsqueeze(1)
        # 初始化匹配后的批次数组
        matched_batch = np.zeros_like(red_channel)
        for i in range(red_channel.shape[0]):
            matched_batch[i] = match_histograms(red_channel[i], green_channel[i])
        # 将匹配后的结果转换为张量并移动到 GPU
        mix_channel = torch.from_numpy(matched_batch).float().to(images.device)
        return mix_channel, blue_channel.to(images.device)

    def forward(self, images):
        # 确保 images 在正确的设备上
        mix_channel, blue_channel = self.process_channels(images)
        rgb_channel = images

        v1 = self.BlueLayerResNet(blue_channel)
        v2 = self.RGBLayerResNet(rgb_channel)
        v3 = self.MixLayerResNet(mix_channel)

        v1 = self.attention16(v1)
        v2 = self.attention32(v2)
        v3 = self.attention64(v3)

        v1 = v1.permute(0, 2, 3, 1)  # [batch_size, H, W, channels]
        v2 = v2.permute(0, 2, 3, 1)
        v3 = v3.permute(0, 2, 3, 1)

        return v1, v2, v3

if __name__ == '__main__':
    from dataset import dataloader
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    train_loader, val_loader = dataloader(batch_size)
    # 在循环外实例化模型
    model = VerticesSelection(device).to(device)
    for images, labels in train_loader:
        # 将 images 移动到 device 上
        images = images.to(device)
        a, b, c = model(images)
        print(a.shape,b.shape,c.shape)
        break