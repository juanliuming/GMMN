import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import set_fusion_strategy


# 指定设备（CPU 或 GPU）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WeightedFusionModel(nn.Module):
    def __init__(self, num_classes,device):
        super(WeightedFusionModel, self).__init__()
        # 调整线性层，输入和输出维度都为128
        # 定义可学习的权重参数，初始化为指定的值
        initial_weights = torch.tensor([0.25, 0.5, 0.25])
        self.weights = nn.Parameter(initial_weights)

        # 融合后的全连接层
        self.fusion_fc = nn.Linear(512, 512)

        # 最终的分类器，输出指定类别数
        self.classifier = nn.Linear(512, out_features=num_classes)

        # 可选的归一化和 Dropout 层
        self.norm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=0.4)
        self.to(device)

    def forward(self, x1, x2, x3):
        # 对每个输入进行线性变换和激活
        x1_proj = x1
        x2_proj = x2
        x3_proj = x3

        # 对节点特征进行最大池化和平均池化，然后求平均
        x1_max = torch.max(x1_proj, dim=1)[0]
        x1_avg = torch.mean(x1_proj, dim=1)
        x1_pooled = (x1_max + x1_avg) / 2

        x2_max = torch.max(x2_proj, dim=1)[0]
        x2_avg = torch.mean(x2_proj, dim=1)
        x2_pooled = (x2_max + x2_avg) / 2

        x3_max = torch.max(x3_proj, dim=1)[0]
        x3_avg = torch.mean(x3_proj, dim=1)
        x3_pooled = (x3_max + x3_avg) / 2

        # 使用 Softmax 对权重进行归一化
        weights = F.softmax(self.weights, dim=0)
        # print(x1_pooled.shape, x2_pooled.shape, x3_pooled.shape)
        # 加权融合特征
        x_fusion = (x1_pooled * weights[0] +
                    x2_pooled * weights[1] +
                    x3_pooled * weights[2])

        # 可选的归一化和 Dropout
        x_fusion = self.norm(x_fusion)
        x_fusion = self.dropout(x_fusion)

        # 通过融合层
        x_fusion = F.relu(self.fusion_fc(x_fusion))

        # 输出层，得到最终的分类结果
        logits = self.classifier(x_fusion)

        return logits

# 示例用法
if __name__ == "__main__":
    # 实例化模型，并移动到指定设备
    num_classes = 3
    model = WeightedFusionModel(num_classes,device).to(device)

    # 创建示例输入数据，并移动到指定设备
    batch_size = 4
    x1 = torch.randn(batch_size, 256, 512).to(device)  # (batch_size, num_nodes1, feature_dim)
    x2 = torch.randn(batch_size, 1024, 512).to(device)
    x3 = torch.randn(batch_size, 4096, 512).to(device)

    # 前向传播，获取预测结果
    logits = model(x1, x2, x3)  # (batch_size, num_classes)
    predictions = torch.argmax(logits, dim=1)  # (batch_size,)

    print("预测类别索引：", predictions)
    print(logits)
