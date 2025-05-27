import torch
import torch.nn as nn
import torch.nn.functional as F

def create_8_connected_adjacency(rows, cols, device):
    N = rows * cols  
    idx_grid = torch.arange(N, device=device).reshape(rows, cols)

    source_indices = []
    target_indices = []

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

    node_indices = torch.arange(N, device=device)
    source_indices.append(node_indices)
    target_indices.append(node_indices)

    source_indices = torch.cat(source_indices)
    target_indices = torch.cat(target_indices)

    indices = torch.stack([source_indices, target_indices], dim=0)
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)

    A_sparse = torch.sparse_coo_tensor(indices, values, (N, N)).coalesce()

    return A_sparse

class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.Tensor(input_dim, output_dim))
        nn.init.kaiming_uniform_(self.W, nonlinearity='relu')
        self.bn = nn.BatchNorm1d(output_dim)

    def forward(self, H, A_normalized):
        batch_size, num_nodes, _ = H.shape
        H = H.reshape(-1, self.input_dim)  # (batch_size * num_nodes, input_dim)
        H_transformed = torch.matmul(H, self.W)  # (batch_size * num_nodes, output_dim)
        H_transformed = self.bn(H_transformed)
        H_transformed = F.relu(H_transformed)
        H_transformed = H_transformed.reshape(batch_size, num_nodes, self.output_dim)
        H_transformed = H_transformed.transpose(0, 1).reshape(num_nodes, -1)  # (num_nodes, batch_size * output_dim)
        H_agg = torch.sparse.mm(A_normalized, H_transformed)  # (num_nodes, batch_size * output_dim)
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
        D = torch.sparse.sum(self.A, dim=1).to_dense()
        D_inv_sqrt = torch.pow(D + 1e-5, -0.5)
        D_inv_sqrt[D_inv_sqrt == float('inf')] = 0.0
        indices = self.A.indices()
        values = self.A.values()
        row_indices = indices[0]
        col_indices = indices[1]

        values = values * D_inv_sqrt[row_indices] * D_inv_sqrt[col_indices]

        self.A_normalized = torch.sparse_coo_tensor(indices, values, self.A.size()).coalesce().to(device)

    def forward(self, batch_image_feature_map, Flatten=False, Reshape=False):
        batch_image_feature_map = batch_image_feature_map.to(self.device)
        if Flatten:
            H = batch_image_feature_map.view(-1, self.rows * self.cols, self.input_dim)
        else:
            H = batch_image_feature_map
        H_next = self.gc_layer(H, self.A_normalized)
        if Reshape:
            H_next = H_next.view(-1, self.rows, self.cols, self.output_dim)
        return H_next

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_image_feature_map = torch.rand(32, 16, 16, 32).to(device)
    gc_batch = GraphConvolutionBatch(16, 16, input_dim=32, output_dim=64, device=device)
    output_feature_map_batch = gc_batch(batch_image_feature_map, Flatten=True, Reshape=False)
    print("Output feature matrix shape:", output_feature_map_batch.shape)
