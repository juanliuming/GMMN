from AdjacencyBlock import GraphConvolutionBatch
from MappingBlock import CrossLayerFeatureUpdate
from mamba_ssm import Mamba

import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GMMNBlock(nn.Module):
    def __init__(self, device):
        super(GMMNBlock, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5, requires_grad=True).to("cuda"))
        self.AdjacencyBlock8x8 = GraphConvolutionBatch(16, 16, 512, 512, device)
        self.AdjacencyBlock16x16 = GraphConvolutionBatch(32, 32, 512, 512, device)
        self.AdjacencyBlock32x32 = GraphConvolutionBatch(64, 64, 512, 512, device)
        self.AdjacencyBlock8x81 = GraphConvolutionBatch(16, 16, 512, 512, device)
        self.AdjacencyBlock16x161 = GraphConvolutionBatch(32, 32, 512, 512, device)
        self.AdjacencyBlock32x321 = GraphConvolutionBatch(64, 64, 512, 512, device)
        self.model_32_16 = CrossLayerFeatureUpdate(512, 32, 16, device=device)
        self.model_64_32 = CrossLayerFeatureUpdate(512, 64, 32, device=device)
        self.Mamba1 = Mamba(d_model=512, d_state=16, d_conv=3, expand=2)
        self.Mamba2 = Mamba(d_model=512, d_state=16, d_conv=3, expand=2)
        self.Mamba3 = Mamba(d_model=512, d_state=16, d_conv=3, expand=2)
        self.Mamba4 = Mamba(d_model=512, d_state=16, d_conv=3, expand=2)
        self.Mamba5 = Mamba(d_model=512, d_state=16, d_conv=3, expand=2)
        self.Mamba6 = Mamba(d_model=512, d_state=16, d_conv=3, expand=2)
        self.device = device
        self.to(device)

    def forward(self, image16x16, image32x32, image64x64):
        v1 = image16x16
        v2 = image32x32
        v3 = image64x64

        # 邻接更新
        v1 = self.AdjacencyBlock8x8(v1, Flatten=True, Reshape=False).to("cuda")
        v2 = self.AdjacencyBlock16x16(v2, Flatten=True, Reshape=False).to("cuda")
        v3 = self.AdjacencyBlock32x32(v3, Flatten=True, Reshape=False).to("cuda")

        v1 = self.Mamba1(v1)
        v2 = self.Mamba2(v2)
        v3 = self.Mamba3(v3)

        v2_new1, v1_new = self.model_32_16(v2, v1)
        v3_new, v2_new2 = self.model_64_32(v3, v2)

        v2_new = self.alpha * v2_new1 + (1 - self.alpha) * v2_new2

        v1_new = self.Mamba4(v1_new)
        v2_new = self.Mamba5(v2_new)
        v3_new = self.Mamba6(v3_new)

        v1_new = self.AdjacencyBlock8x81(v1_new, Flatten=False, Reshape=False).to("cuda")
        v2_new = self.AdjacencyBlock16x161(v2_new, Flatten=False, Reshape=False).to("cuda")
        v3_new = self.AdjacencyBlock32x321(v3_new, Flatten=False, Reshape=False).to("cuda")

        return v1_new, v2_new, v3_new
        # return v1, v2, v3


if __name__ == '__main__':
    v1 = torch.randn((4, 16, 16, 512)).to("cuda")
    v2 = torch.randn((4, 32, 32, 512)).to("cuda")
    v3 = torch.randn((4, 64, 64, 512)).to("cuda")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GMMNBlock(device=device).to(device)
    a, b, c = model.forward(v1, v2, v3)
    print(a.shape, b.shape, c.shape)