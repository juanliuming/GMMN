import torch
import torch.nn as nn
import torch.nn.functional as F

def create_8_connected_adjacency(rows, cols, device):
    """
    创建八连通邻接矩阵的稀疏表示。
    """
    N = rows * cols  # 节点总数
    idx_grid = torch.arange(N, device=device).reshape(rows, cols)

    source_indices = []
    target_indices = []

    # 八个方向的偏移量
    directions = [(-1, -1), (-1, 0), (-1, 1),
                  (0, -1),          (0, 1),
                  (1, -1),  (1, 0), (1, 1)]

    for dx, dy in directions:
        if dx == -1:
            source_slice_i = slice(1, rows)
            target_slice_i = slice(0, rows - 1)
        elif dx == 0:
            source_slice_i = slice(0, rows)
            target_slice_i = slice(0, rows)
        elif dx == 1:
            source_slice_i = slice(0, rows - 1)
            target_slice_i = slice(1, rows)

        if dy == -1:
            source_slice_j = slice(1, cols)
            target_slice_j = slice(0, cols - 1)
        elif dy == 0:
            source_slice_j = slice(0, cols)
            target_slice_j = slice(0, cols)
        elif dy == 1:
            source_slice_j = slice(0, cols - 1)
            target_slice_j = slice(1, cols)

        source_idx = idx_grid[source_slice_i, source_slice_j].reshape(-1)
        target_idx = idx_grid[target_slice_i, target_slice_j].reshape(-1)

        source_indices.append(source_idx)
        target_indices.append(target_idx)

    # 添加自连接
    node_indices = torch.arange(N, device=device)
    source_indices.append(node_indices)
    target_indices.append(node_indices)

    # 合并所有索引
    source_indices = torch.cat(source_indices)
    target_indices = torch.cat(target_indices)

    # 创建边索引和对应的值
    indices = torch.stack([source_indices, target_indices], dim=0)
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)

    # 创建稀疏的邻接矩阵
    A_sparse = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()

    return A_sparse

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        初始化图卷积层，使用可学习的权重矩阵。
        :param input_dim: 输入特征维度 (例如 32)
        :param output_dim: 输出特征维度 (例如 64)
        """
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.Tensor(input_dim, output_dim))
        # 使用 Kaiming 初始化
        nn.init.kaiming_uniform_(self.W, nonlinearity='relu')
        # 添加批归一化层
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, H, A_normalized):
        """
        前向传播过程，执行图卷积操作。
        :param H: 输入特征矩阵 (形状: [batch_size, num_nodes, input_dim])
        :param A_normalized: 归一化后的邻接矩阵 (稀疏格式，形状: [num_nodes, num_nodes])
        :return: 输出特征矩阵 (形状: [batch_size, num_nodes, output_dim])
        """
        batch_size, num_nodes, _ = H.shape
        # 先进行特征变换
        H = H.reshape(-1, self.input_dim)  # (batch_size * num_nodes, input_dim)
        H_transformed = torch.matmul(H, self.W)  # (batch_size * num_nodes, output_dim)
        # 批归一化
        H_transformed = self.bn(H_transformed)
        # 激活函数
        H_transformed = F.relu(H_transformed)
        # 重新调整形状
        H_transformed = H_transformed.reshape(batch_size, num_nodes, self.output_dim)
        # 将批次和节点维度合并，以便进行稀疏矩阵乘法
        H_transformed = H_transformed.transpose(0, 1).reshape(num_nodes, -1)  # (num_nodes, batch_size * output_dim)
        # 聚合邻居特征
        H_agg = torch.sparse.mm(A_normalized, H_transformed)  # (num_nodes, batch_size * output_dim)
        # 还原形状
        H_agg = H_agg.reshape(num_nodes, batch_size, self.output_dim).transpose(0, 1)  # (batch_size, num_nodes, output_dim)
        return H_agg

class GraphConvolutionBatch(nn.Module):
    def __init__(self, rows, cols, input_dim=32, output_dim=32, device='cuda'):
        super(GraphConvolutionBatch, self).__init__()
        self.rows = rows
        self.cols = cols
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.gc_layer = GraphConvolution(input_dim, output_dim).to(device)
        self.A = create_8_connected_adjacency(rows, cols, device)
        # 对邻接矩阵进行对称归一化
        # 计算度矩阵 D
        D = torch.sparse.sum(self.A, dim=1).to_dense()
        D_inv_sqrt = torch.pow(D + 1e-5, -0.5)
        D_inv_sqrt[D_inv_sqrt == float('inf')] = 0.0

        # 获取邻接矩阵的索引和数值
        indices = self.A.indices()
        values = self.A.values()
        row_indices = indices[0]
        col_indices = indices[1]

        # 对邻接矩阵的非零值进行归一化
        values = values * D_inv_sqrt[row_indices] * D_inv_sqrt[col_indices]

        # 创建归一化后的邻接矩阵
        self.A_normalized = torch.sparse_coo_tensor(indices, values, self.A.size()).coalesce().to(device)

    def forward(self, batch_image_feature_map, Flatten=False, Reshape=False):
        batch_image_feature_map = batch_image_feature_map.to(self.device)
        # 将图像特征展平为适合图卷积的形状 (batch_size, num_nodes, input_dim)
        if Flatten:
            H = batch_image_feature_map.view(-1, self.rows * self.cols, self.input_dim)
        else:
            H = batch_image_feature_map
        # 执行图卷积操作
        H_next = self.gc_layer(H, self.A_normalized)
        # 如果需要，重塑输出特征矩阵为原始的二维形状
        if Reshape:
            H_next = H_next.view(-1, self.rows, self.cols, self.output_dim)
        return H_next

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 创建一个批次大小为 2 的随机图像特征输入 (2, 16, 16, 32)
    batch_image_feature_map = torch.rand(32, 16, 16, 32).to(device)
    # 创建图卷积批次模块并调用，设置 output_dim=64
    gc_batch = GraphConvolutionBatch(16, 16, input_dim=32, output_dim=64, device=device)
    output_feature_map_batch = gc_batch(batch_image_feature_map, Flatten=True, Reshape=False)
    # 输出特征矩阵的形状
    print("Output feature matrix shape:", output_feature_map_batch.shape)
