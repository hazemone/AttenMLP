import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_activation(activation):
    if activation.lower() == 'gelu':
        return nn.GELU()
    elif activation.lower() == 'rrelu':
        return nn.RReLU(inplace=True)
    elif activation.lower() == 'selu':
        return nn.SELU(inplace=True)
    elif activation.lower() == 'silu':
        return nn.SiLU(inplace=True)
    elif activation.lower() == 'hardswish':
        return nn.Hardswish(inplace=True)
    elif activation.lower() == 'leakyrelu':
        return nn.LeakyReLU(inplace=True)
    else:
        return nn.ReLU(inplace=True)

def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

try:
    from pointnet2_ops import pointnet2_utils as _pn2
    _CUDA_FPS = True
except ImportError:
    _CUDA_FPS = False

def farthest_point_sample(xyz, npoint):
    if _CUDA_FPS and xyz.is_cuda:
        return _pn2.furthest_point_sample(xyz.float().contiguous(), npoint).long()
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def knn_point(nsample, xyz, new_xyz):
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(1, max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[:d_model//2])
        self.register_buffer('pe', pe)

    def forward(self, x):
        # Detect format by checking whether the last dim matches d_model:
        #   B x N x C: x.size(-1) == d_model  -> add PE along sequence dim 1
        #   B x C x N: x.size(-1) == N (not d_model) -> add PE along sequence dim 2
        if x.size(-1) == self.pe.size(-1):  # B x N x C format
            return x + self.pe[:, :x.size(1), :]
        else:  # B x C x N format
            return x + self.pe.permute(0, 2, 1)[:, :, :x.size(-1)]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Initialize weights using Xavier initialization
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights

    def forward(self, x, mask=None):
        batch_size = x.size(0)

        # Add positional encoding and save for residual (pre-LN design)
        x = self.pos_encoding(x)
        residual = x

        # Apply layer normalization before projections (Pre-LN transformer)
        x = self.layer_norm(x)

        # Linear projections
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into multiple heads
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        output, attention_weights = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and apply final linear transformation
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.dropout(self.W_o(output))

        # Residual connection from pre-LN input (correct Pre-LN design)
        output = output + residual

        return output

# Rest of the classes remain the same
class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center", **kwargs):
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()

        fps_idx = farthest_point_sample(xyz, self.groups).long()
        new_xyz = index_points(xyz, fps_idx)
        new_points = index_points(points, fps_idx)

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_points = index_points(points, idx)
        
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize == "anchor":
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(dim=-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)
            grouped_points = self.affine_alpha * grouped_points + self.affine_beta

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points

class ConvBNReLU1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, activation='relu'):
        super(ConvBNReLU1D, self).__init__()
        self.act = get_activation(activation)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, bias=bias),
            nn.BatchNorm1d(out_channels),
            self.act
        )

    def forward(self, x):
        return self.net(x)

class ConvBNReLURes1D(nn.Module):
    def __init__(self, channel, kernel_size=1, groups=1, res_expansion=1.0, bias=True, activation='relu'):
        super(ConvBNReLURes1D, self).__init__()
        self.act = get_activation(activation)
        self.net1 = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=int(channel * res_expansion),
                      kernel_size=kernel_size, groups=groups, bias=bias),
            nn.BatchNorm1d(int(channel * res_expansion)),
            self.act
        )
        if groups > 1:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, groups=groups, bias=bias),
                nn.BatchNorm1d(channel),
                self.act,
                nn.Conv1d(in_channels=channel, out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel),
            )
        else:
            self.net2 = nn.Sequential(
                nn.Conv1d(in_channels=int(channel * res_expansion), out_channels=channel,
                          kernel_size=kernel_size, bias=bias),
                nn.BatchNorm1d(channel)
            )

    def forward(self, x):
        return self.act(self.net2(self.net1(x)) + x)

class PreExtraction(nn.Module):
    def __init__(self, channels, out_channels, blocks=1, groups=1, res_expansion=1, bias=True,
                 activation='relu', use_xyz=True, num_heads=16):
        super(PreExtraction, self).__init__()
        in_channels = 3 + 2 * channels if use_xyz else 2 * channels
        self.transfer = ConvBNReLU1D(in_channels, out_channels, bias=bias, activation=activation)
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(out_channels, groups=groups, res_expansion=res_expansion,
                                bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)
        self.attention = MultiHeadAttention(out_channels, num_heads)

    def forward(self, x):
        b, n, s, d = x.size()
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        x = self.transfer(x)
        batch_size, _, _ = x.size()
        x = self.operation(x)
        
        x = x.permute(0, 2, 1)
        x = self.attention(x)
        x = x.permute(0, 2, 1)
        
        x = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        return x

class PosExtraction(nn.Module):
    def __init__(self, channels, blocks=1, groups=1, res_expansion=1, bias=True, activation='relu', num_heads=4):
        super(PosExtraction, self).__init__()
        operation = []
        for _ in range(blocks):
            operation.append(
                ConvBNReLURes1D(channels, groups=groups, res_expansion=res_expansion, bias=bias, activation=activation)
            )
        self.operation = nn.Sequential(*operation)
        self.attention = MultiHeadAttention(channels, num_heads)

    def forward(self, x):
        x = self.operation(x)
        x = x.permute(0, 2, 1)
        x = self.attention(x)
        x = x.permute(0, 2, 1)
        return x

class Model(nn.Module):
    def __init__(self, points=1024, class_num=40, embed_dim=64, groups=1, res_expansion=1.0,
                 activation="relu", bias=True, use_xyz=True, normalize="center",
                 dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                 k_neighbors=[32, 32, 32, 32], reducers=[2, 2, 2, 2], num_heads=4, parser_args=None,
                 dropout=0.1):
        super(Model, self).__init__()
        self.stages = len(pre_blocks)
        self.class_num = class_num
        self.points = points
        self.args = parser_args
        
        # Combine embedding and positional encoding in a sequential manner
        self.input_embedding = nn.Sequential(
            ConvBNReLU1D(3, embed_dim, bias=bias, activation=activation),
            PositionalEncoding(embed_dim)
        )

        
        assert len(pre_blocks) == len(k_neighbors) == len(reducers) == len(pos_blocks) == len(dim_expansion), \
            "Please check stage number consistent for pre_blocks, pos_blocks k_neighbors, reducers."
        
        self.local_grouper_list = nn.ModuleList()
        self.pre_blocks_list = nn.ModuleList()
        self.pos_blocks_list = nn.ModuleList()
        last_channel = embed_dim
        
        if self.args is not None:
            # Ratio of sparse-net channels to dense-net channels at each stage.
            # Dense embed_dim == args.num_channel; sparse embed_dim == this model's embed_dim.
            self.factor_channel = max(1, int(embed_dim / self.args.num_channel))
            
        anchor_points = self.points
        for i in range(len(pre_blocks)):
            out_channel = last_channel * dim_expansion[i]
            pre_block_num = pre_blocks[i]
            pos_block_num = pos_blocks[i]
            kneighbor = k_neighbors[i]
            reduce = reducers[i]
            anchor_points = anchor_points // reduce
            
            # Enhanced local grouper with normalized coordinates
            local_grouper = LocalGrouper(last_channel, anchor_points, kneighbor, use_xyz, normalize)
            self.local_grouper_list.append(local_grouper)
            
            # Pre-extraction blocks with attention
            pre_block_module = PreExtraction(last_channel, out_channel, pre_block_num, groups=groups,
                                          res_expansion=res_expansion, bias=bias, activation=activation,
                                          use_xyz=use_xyz, num_heads=num_heads)
            self.pre_blocks_list.append(pre_block_module)
            
            # Position extraction blocks with attention
            pos_block_module = PosExtraction(out_channel, pos_block_num, groups=groups,
                                          res_expansion=res_expansion, bias=bias, activation=activation,
                                          num_heads=num_heads)
            self.pos_blocks_list.append(pos_block_module)
            
            last_channel = out_channel
            
        if self.args is not None:
            self.maxpool = torch.nn.MaxPool1d(int(self.args.num_points_high / self.args.num_points_low))
            
        self.act = get_activation(activation)
        
        # Enhanced classifier with dropout and batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(last_channel, 512),
            nn.BatchNorm1d(512),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            self.act,
            nn.Dropout(dropout),
            nn.Linear(256, self.class_num)
        )

    def forward(self, x, inter_x_prv=None, debug=False):
        inter_x = []
        xyz = x.permute(0, 2, 1)
        batch_size, _, _ = x.size()
        
        # Use combined input embedding
        x = self.input_embedding(x)

        for i in range(self.stages):
            # Local grouping
            xyz, x = self.local_grouper_list[i](xyz, x.permute(0, 2, 1))
            if debug:
                print(f"Stage {i} after local grouping:", xyz.shape, x.size())
                
            # Pre-extraction with attention
            x = self.pre_blocks_list[i](x)
            if debug:
                print(f"Stage {i} after pre-extraction:", x.shape)
                
            # Position extraction with attention
            x = self.pos_blocks_list[i](x)
            
            # Apply intermediate feature fusion if available
            if inter_x_prv is not None:
                x_pool = self.maxpool(inter_x_prv[i])
                x_pool = x_pool.repeat(1, self.factor_channel, 1)
                x = x_pool * x
                
            if debug:
                print(f"Stage {i} after pos-extraction:", x.shape)
                
            inter_x.append(x)

        # Global feature aggregation
        x = F.adaptive_max_pool1d(x, 1).squeeze(dim=-1)
        inter_x.append(x)
        
        if debug:
            print('END OF DEBUG!')

        # Feature fusion for final classification
        if inter_x_prv is not None:
            if self.args.last_layer_concat == 'multiply':
                x_pool = inter_x_prv[-1].repeat(1, self.factor_channel)
                x = x_pool * x
            elif self.args.last_layer_concat == 'add':
                x_pool = inter_x_prv[-1].repeat(1, self.factor_channel)
                x = x_pool * self.args.add_factor + x
            elif self.args.last_layer_concat == 'concat':
                x = torch.cat((x, inter_x_prv[-1]), -1)

        # Final classification
        x = self.classifier(x)
        return x, inter_x


def pointMLP(num_classes=40, points=1024, k_neighbor=[24, 24, 24, 24], num_channel=64, num_heads=8, dropout=0.1, **kwargs):
    return Model(points=points, class_num=num_classes, embed_dim=64, groups=1, res_expansion=1.0,
                activation="relu", bias=False, use_xyz=False, normalize="anchor",
                dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                k_neighbors=k_neighbor, reducers=[2, 2, 2, 2], num_heads=num_heads, dropout=dropout, **kwargs)


def pointMLPElite(num_classes=40, num_heads=8, dropout=0.1, **kwargs):
    return Model(points=1024, class_num=num_classes, embed_dim=32, groups=1, res_expansion=0.25,
                activation="relu", bias=False, use_xyz=False, normalize="anchor",
                dim_expansion=[2, 2, 2, 1], pre_blocks=[1, 1, 2, 1], pos_blocks=[1, 1, 2, 1],
                k_neighbors=[24, 24, 24, 24], reducers=[2, 2, 2, 2], num_heads=num_heads,
                dropout=dropout, **kwargs)


if __name__ == '__main__':
    data = torch.rand(2, 3, 1024)
    print("===> testing pointMLP ...")
    model = pointMLP()
    out = model(data)
    print(out[0].shape)  # Classification output