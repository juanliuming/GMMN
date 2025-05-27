import torch
import torch.nn as nn
import torch.nn.functional as F

def create_cross_layer_adjacency(orig_size, downsampled_size, device):
    factor = orig_size // downsampled_size  # 降采样比例
    # 生成降采样层的节点索引 (M)
    down_i = torch.arange(downsampled_size, device=device)
    down_j = torch.arange(downsampled_size, device=device)
    down_i, down_j = torch.meshgrid(down_i, down_j, indexing='ij')
    down_node_idx = (down_i * downsampled_size + down_j).reshape(-1)  # (M,)

    # 对应的原始图层的节点索引 (N)
    orig_i = down_i * factor
    orig_j = down_j * factor

    orig_node_idx_1 = (orig_i * orig_size + orig_j).reshape(-1)
    orig_node_idx_2 = (orig_i * orig_size + orig_j + 1).reshape(-1)
    orig_node_idx_3 = ((orig_i + 1) * orig_size + orig_j).reshape(-1)
    orig_node_idx_4 = ((orig_i + 1) * orig_size + orig_j + 1).reshape(-1)

    # 合并所有原始节点索引
    orig_node_indices = torch.cat([orig_node_idx_1, orig_node_idx_2, orig_node_idx_3, orig_node_idx_4], dim=0)
    # 对应的降采样节点索引重复4次
    down_node_indices = down_node_idx.repeat(4)

    # 创建边索引和对应的值
    indices = torch.stack([down_node_indices, orig_node_indices], dim=0)
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)

    size = (downsampled_size ** 2, orig_size ** 2)
    A_cross = torch.sparse_coo_tensor(indices, values, size, device=device).coalesce()

    # 对邻接矩阵进行对称归一化
    # 计算度矩阵
    deg_orig = torch.sparse.sum(A_cross.transpose(0, 1), dim=1).to_dense()
    deg_down = torch.sparse.sum(A_cross, dim=1).to_dense()

    # 计算度的倒数平方根
    deg_orig_inv_sqrt = torch.pow(deg_orig, -0.5)
    deg_orig_inv_sqrt[deg_orig_inv_sqrt == float('inf')] = 0

    deg_down_inv_sqrt = torch.pow(deg_down, -0.5)
    deg_down_inv_sqrt[deg_down_inv_sqrt == float('inf')] = 0

    # 创建度的倒数平方根对角矩阵
    D_orig_inv_sqrt = torch.diag(deg_orig_inv_sqrt)
    D_down_inv_sqrt = torch.diag(deg_down_inv_sqrt)

    # 将度矩阵应用于邻接矩阵，实现对称归一化
    A_cross = torch.sparse.mm(D_down_inv_sqrt, torch.sparse.mm(A_cross, D_orig_inv_sqrt))

    return A_cross

class CrossLayerFeatureUpdate(nn.Module):
    """
    跨层特征更新模型，使用归一化的邻接矩阵和线性变换更新特征。
    """
    def __init__(self, feature_dim, orig_size, downsampled_size, device):
        super(CrossLayerFeatureUpdate, self).__init__()
        # 定义原始图层到降采样图层的线性变换
        self.W_orig_to_new = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_new_to_orig = nn.Linear(feature_dim, feature_dim, bias=False)

        # 使用 Xavier 初始化
        nn.init.xavier_uniform_(self.W_orig_to_new.weight)
        nn.init.xavier_uniform_(self.W_new_to_orig.weight)

        # 创建跨层归一化邻接矩阵（稀疏格式）
        self.A_cross = create_cross_layer_adjacency(orig_size, downsampled_size, device)

        # 添加 Layer Normalization 层
        self.ln_orig_to_new = nn.LayerNorm(feature_dim)
        self.ln_new_to_orig = nn.LayerNorm(feature_dim)

    def forward(self, H_orig, H_down):
        """
        前向传播，更新原始图层和降采样图层的特征。
        参数:
        - H_orig: 原始图层的特征矩阵 (batch_size, N, feature_dim)
        - H_down: 降采样图层的特征矩阵 (batch_size, M, feature_dim)
        返回:
        - 更新后的降采样图层特征 H_down_updated
        - 更新后的原始图层特征 H_high_updated
        """
        batch_size = H_orig.size(0)

        # 从原始图层传递到降采样图层
        H_new_updated = torch.sparse.mm(self.A_cross, H_orig.transpose(0, 1).reshape(H_orig.size(1), -1))  # (M, batch_size * feature_dim)
        H_new_updated = H_new_updated.reshape(H_down.size(1), batch_size, -1).transpose(0, 1)  # (batch_size, M, feature_dim)
        H_new_updated = self.W_orig_to_new(H_new_updated)
        H_new_updated = self.ln_orig_to_new(H_new_updated)
        H_new_updated = F.relu(H_new_updated)

        # 从降采样图层传递回原始图层
        H_orig_updated = torch.sparse.mm(self.A_cross.transpose(0, 1), H_down.transpose(0, 1).reshape(H_down.size(1), -1))  # (N, batch_size * feature_dim)
        H_orig_updated = H_orig_updated.reshape(H_orig.size(1), batch_size, -1).transpose(0, 1)  # (batch_size, N, feature_dim)
        H_orig_updated = self.W_new_to_orig(H_orig_updated)
        H_orig_updated = self.ln_new_to_orig(H_orig_updated)
        H_orig_updated = F.relu(H_orig_updated)

        # H_new_updated = H_down + H_new_updated
        # H_orig_updated = H_orig + H_orig_updated
        return H_orig_updated, H_new_updated

if __name__ == '__main__':
    # 设置设备为 cuda 或 cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 设置参数
    orig_size = 64  # 原始图层是64x64
    downsampled_size = 32  # 降采样图层是32x32
    feature_dim = 64  # 每个节点的特征维度
    batch_size = 2  # 批次大小

    # 输入特征矩阵 (batch_size, N, feature_dim)
    H_orig = torch.randn(batch_size, orig_size ** 2, feature_dim).to(device)  # 原始图层的特征矩阵
    H_down = torch.randn(batch_size, downsampled_size ** 2, feature_dim).to(device)  # 降采样图层的特征矩阵

    # 创建并移动模型到指定设备
    model = CrossLayerFeatureUpdate(feature_dim, orig_size, downsampled_size, device).to(device)

    # 执行前向传播，更新特征
    H_high_updated, H_down_updated = model(H_orig, H_down)

    # 输出更新后的特征
    print("原始高尺寸更新:", H_high_updated.shape)
    print("降采样尺寸更新:", H_down_updated.shape)
