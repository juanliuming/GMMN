import torch
import torch.nn as nn
import torch.nn.functional as F

def create_cross_layer_adjacency(orig_size, downsampled_size, device):
    factor = orig_size // downsampled_size 

    down_i = torch.arange(downsampled_size, device=device)
    down_j = torch.arange(downsampled_size, device=device)
    down_i, down_j = torch.meshgrid(down_i, down_j, indexing='ij')
    down_node_idx = (down_i * downsampled_size + down_j).reshape(-1)  # (M,)

    orig_i = down_i * factor
    orig_j = down_j * factor

    orig_node_idx_1 = (orig_i * orig_size + orig_j).reshape(-1)
    orig_node_idx_2 = (orig_i * orig_size + orig_j + 1).reshape(-1)
    orig_node_idx_3 = ((orig_i + 1) * orig_size + orig_j).reshape(-1)
    orig_node_idx_4 = ((orig_i + 1) * orig_size + orig_j + 1).reshape(-1)

    orig_node_indices = torch.cat([orig_node_idx_1, orig_node_idx_2, orig_node_idx_3, orig_node_idx_4], dim=0)

    down_node_indices = down_node_idx.repeat(4)

    indices = torch.stack([down_node_indices, orig_node_indices], dim=0)
    values = torch.ones(indices.shape[1], dtype=torch.float32, device=device)

    size = (downsampled_size ** 2, orig_size ** 2)
    A_cross = torch.sparse_coo_tensor(indices, values, size, device=device).coalesce()


    deg_orig = torch.sparse.sum(A_cross.transpose(0, 1), dim=1).to_dense()
    deg_down = torch.sparse.sum(A_cross, dim=1).to_dense()


    deg_orig_inv_sqrt = torch.pow(deg_orig, -0.5)
    deg_orig_inv_sqrt[deg_orig_inv_sqrt == float('inf')] = 0

    deg_down_inv_sqrt = torch.pow(deg_down, -0.5)
    deg_down_inv_sqrt[deg_down_inv_sqrt == float('inf')] = 0

    D_orig_inv_sqrt = torch.diag(deg_orig_inv_sqrt)
    D_down_inv_sqrt = torch.diag(deg_down_inv_sqrt)

    A_cross = torch.sparse.mm(D_down_inv_sqrt, torch.sparse.mm(A_cross, D_orig_inv_sqrt))

    return A_cross

class CrossLayerFeatureUpdate(nn.Module):

    def __init__(self, feature_dim, orig_size, downsampled_size, device):
        super(CrossLayerFeatureUpdate, self).__init__()

        self.W_orig_to_new = nn.Linear(feature_dim, feature_dim, bias=False)
        self.W_new_to_orig = nn.Linear(feature_dim, feature_dim, bias=False)

        nn.init.xavier_uniform_(self.W_orig_to_new.weight)
        nn.init.xavier_uniform_(self.W_new_to_orig.weight)

        self.A_cross = create_cross_layer_adjacency(orig_size, downsampled_size, device)

        self.ln_orig_to_new = nn.LayerNorm(feature_dim)
        self.ln_new_to_orig = nn.LayerNorm(feature_dim)

    def forward(self, H_orig, H_down):

        batch_size = H_orig.size(0)

        H_new_updated = torch.sparse.mm(self.A_cross, H_orig.transpose(0, 1).reshape(H_orig.size(1), -1))  # (M, batch_size * feature_dim)
        H_new_updated = H_new_updated.reshape(H_down.size(1), batch_size, -1).transpose(0, 1)  # (batch_size, M, feature_dim)
        H_new_updated = self.W_orig_to_new(H_new_updated)
        H_new_updated = self.ln_orig_to_new(H_new_updated)
        H_new_updated = F.relu(H_new_updated)

        H_orig_updated = torch.sparse.mm(self.A_cross.transpose(0, 1), H_down.transpose(0, 1).reshape(H_down.size(1), -1))  # (N, batch_size * feature_dim)
        H_orig_updated = H_orig_updated.reshape(H_orig.size(1), batch_size, -1).transpose(0, 1)  # (batch_size, N, feature_dim)
        H_orig_updated = self.W_new_to_orig(H_orig_updated)
        H_orig_updated = self.ln_new_to_orig(H_orig_updated)
        H_orig_updated = F.relu(H_orig_updated)

        # H_new_updated = H_down + H_new_updated
        # H_orig_updated = H_orig + H_orig_updated
        return H_orig_updated, H_new_updated
