import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import set_fusion_strategy


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class WeightedFusionModel(nn.Module):
    def __init__(self, num_classes,device):
        super(WeightedFusionModel, self).__init__()
        initial_weights = torch.tensor([0.25, 0.5, 0.25])
        self.weights = nn.Parameter(initial_weights)

        self.fusion_fc = nn.Linear(512, 512)

        self.classifier = nn.Linear(512, out_features=num_classes)

        self.norm = nn.BatchNorm1d(512)
        self.dropout = nn.Dropout(p=0.4)
        self.to(device)

    def forward(self, x1, x2, x3):
        x1_proj = x1
        x2_proj = x2
        x3_proj = x3

        x1_max = torch.max(x1_proj, dim=1)[0]
        x1_avg = torch.mean(x1_proj, dim=1)
        x1_pooled = (x1_max + x1_avg) / 2

        x2_max = torch.max(x2_proj, dim=1)[0]
        x2_avg = torch.mean(x2_proj, dim=1)
        x2_pooled = (x2_max + x2_avg) / 2

        x3_max = torch.max(x3_proj, dim=1)[0]
        x3_avg = torch.mean(x3_proj, dim=1)
        x3_pooled = (x3_max + x3_avg) / 2

        weights = F.softmax(self.weights, dim=0)
        x_fusion = (x1_pooled * weights[0] +
                    x2_pooled * weights[1] +
                    x3_pooled * weights[2])

        x_fusion = self.norm(x_fusion)
        x_fusion = self.dropout(x_fusion)

        x_fusion = F.relu(self.fusion_fc(x_fusion))

        logits = self.classifier(x_fusion)

        return logits

