import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

def XNorm(x, gamma):
    norm_tensor = torch.norm(x, 4, -1, True)
    return x * gamma / norm_tensor

class UFOAttention(nn.Module):
    '''
    UFO-ViT: High Performance Linear Vision Transformer without Softmax
    https://arxiv.org/abs/2110.07641
    '''
    def __init__(self, embed_size, heads, dropout):

        super(UFOAttention, self).__init__()

        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        self.fc_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_o = nn.Linear(heads * self.head_dim, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.gamma = nn.Parameter(torch.randn((1, heads, 1, 1)))

    def forward(self, queries, keys, values):
        # b_s, nq = queries.shape[:2]
        # nk = keys.shape[1]
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        # q = self.fc_q(queries).view(b_s, nq, self.heads, self.head_dim).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        # k = self.fc_k(keys).view(b_s, nk, self.heads, self.head_dim).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        # v = self.fc_v(values).view(b_s, nk, self.heads, self.head_dim).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.heads, self.head_dim)
        v = self.fc_v(values).permute(0, 2, 1, 3)  # (N, query_len, heads, heads_dim)
        k = self.fc_k(keys).permute(0, 2, 3, 1)
        q = self.fc_q(queries).permute(0, 2, 1, 3)

        kv = torch.matmul(k, v)          # bs,h,c,c
        kv_norm = XNorm(kv, self.gamma)  # bs,h,c,c
        q_norm = XNorm(q, self.gamma)    # bs,h,n,c
        out = torch.matmul(q_norm, kv_norm).permute(0, 2, 1, 3).contiguous().view(N, query_len, self.heads * self.head_dim)
        out = self.fc_o(out)  # (b_s, nq, d_model)

        return out

class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed Size needs to be Divisible by Heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

        self.softmax = nn.Softmax()

    def forward(self, values, keys, query):

        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        """Split Embedding into self.head pieces"""
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)#(N, query_len, heads, heads_dim)
        keys = self.keys(keys)
        queries = self.queries(queries)

        atten = torch.einsum("nqhd, nkhd->nhqk", [queries, keys])
        atten = self.softmax(atten)
        out = torch.einsum("nhql, nlhd->nqhd", [atten, values]).reshape(N, query_len, self.heads * self.head_dim)# attention shape: (N, heads, query_len, key_len)
        out = self.fc_out(out)

        return out

class UFOSelfAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        super(UFOSelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed Size needs to be Divisible by Heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

        self.dropout = nn.Dropout(dropout)

    def forward(self, values, keys, query):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)
        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.matmul(queries, keys.transpose(-2, -1))

        energy_norm = nn.LayerNorm(energy.size()[-1]).to(energy.device)(energy)
        energy_norm = self.dropout(energy_norm)
        values_norm = nn.LayerNorm(values.size()[-1]).to(values.device)(values)
        values_norm = self.dropout(values_norm)
        out = torch.matmul(energy_norm, values_norm).reshape(N, query_len, self.heads * self.head_dim)
        out = self.fc_out(out)

        return out


class SQLayer1d(nn.Module):
    def __init__(self, embed_size):
        super(SQLayer1d, self).__init__()
        self.fc = nn.Sequential(
            # nn.Linear(embed_size, embed_size * 2),
            # nn.ReLU()
            nn.Linear(embed_size, embed_size),
            nn.SiLU(),
        )
    def forward(self, x):
        y = self.fc(x)
        return x * y.expand_as(x)


class Pooling(nn.Module):
    def __init__(self, pool_size=3):
        super().__init__()
        self.avg_pool = nn.AvgPool1d(pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)
        self.max_pool = nn.MaxPool1d(pool_size, stride=1, padding=pool_size // 2)

    def forward(self, x):
        avg_pooled = self.avg_pool(x)
        max_pooled = self.max_pool(x)
        return (avg_pooled + max_pooled) / 2 - x  # Combining pooled differences to highlight changes


class LayerNormChannel(nn.Module):
    """
    LayerNorm only for Channel Dimension.
    Input: tensor in shape [B, h, C]
    """
    def    __init__(self, num_channels, eps=1e-05):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)

        s = (x - u).pow(2).mean(1, keepdim=True)# axis =1 ：压缩列，对各行求均值，返回 m *1 矩阵
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight.unsqueeze(0) * x + self.bias.unsqueeze(0)
        return x
class CrossAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.3):
        """
        Multi-head interactive attention mechanism
        :param embed_size
        :param heads
        :param dropout
        """
        super(CrossAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"

        # Q、K、V
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, value, key, query):
        """
        :param value: Feature A (batch_size, seq_len, embed_size)
        :param key: Feature B (batch_size, seq_len, embed_size)
        :param query: Feature B (batch_size, seq_len, embed_size)
        """
        N, seq_len, embed_size = query.shape

        values = self.values(value).view(N, seq_len, self.heads, self.head_dim)
        keys = self.keys(key).view(N, seq_len, self.heads, self.head_dim)
        queries = self.queries(query).view(N, seq_len, self.heads, self.head_dim)

        #  similarity of interactive attention
        attention_scores = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # the weighted value of interactive attention
        out = torch.einsum("nhqk,nkhd->nqhd", [attention_weights, values])
        out = out.reshape(N, seq_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out

class VariationalAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.3):
        """
        Variational Attention Mechanism
        :param embed_size
        :param heads
        :param dropout
        """
        super(VariationalAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.mu = nn.Parameter(torch.zeros(1, heads, 1, 1))
        self.sigma = nn.Parameter(torch.ones(1, heads, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, value, key, query):
        N, seq_len, embed_size = query.shape

        values = self.values(value).view(N, seq_len, self.heads, self.head_dim)
        keys = self.keys(key).view(N, seq_len, self.heads, self.head_dim)
        queries = self.queries(query).view(N, seq_len, self.heads, self.head_dim)

        attention_scores = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / self.scale
        # Variational inference generates random attention weights
        epsilon = torch.randn_like(self.mu)
        variational_weights = self.mu + self.sigma * epsilon
        attention_scores = attention_scores * variational_weights
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        # output
        out = torch.einsum("nhqk,nkhd->nqhd", [attention_weights, values])
        out = out.reshape(N, seq_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out


class RelativePositionEncodingAttention(nn.Module):
    def __init__(self, embed_size, heads, max_relative_position=10, dropout=0.3):
        super(RelativePositionEncodingAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embed Size must be divisible by Heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.relative_position_embeddings = nn.Parameter(
            torch.randn(max_relative_position * 2 + 1, self.head_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, values, keys, queries):
        N, value_len, key_len, query_len = queries.shape[0], values.shape[1], keys.shape[1], queries.shape[1]

        # reshape
        values = self.values(values).view(N, value_len, self.heads, self.head_dim)
        keys = self.keys(keys).view(N, key_len, self.heads, self.head_dim)
        queries = self.queries(queries).view(N, query_len, self.heads, self.head_dim)

        attention_scores = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / self.scale
        relative_position_bias = self._relative_position_bias(query_len, key_len).to(attention_scores.device)
        attention_scores += relative_position_bias
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        # (N, query_len, heads * head_dim)
        out = torch.einsum("nhqk,nkhd->nqhd", [attention_weights, values]).reshape(N, query_len,
                                                                                   self.heads * self.head_dim)
        out = self.fc_out(out)
        return out

    def _relative_position_bias(self, query_len, key_len):
        # relative position index matrix
        range_vec = torch.arange(query_len)
        relative_positions = range_vec[:, None] - range_vec[None, :]  # (query_len, key_len)
        # [-max_relative_position, max_relative_position]
        max_relative_position = (self.relative_position_embeddings.size(0) - 1) // 2
        relative_positions = relative_positions.clamp(-max_relative_position, max_relative_position)

        # Offset so that the index is non-negative
        relative_positions += max_relative_position
        # (heads, query_len, key_len)
        relative_position_bias = self.relative_position_embeddings[relative_positions]  # (query_len, key_len, head_dim)
        relative_position_bias = relative_position_bias.permute(2, 0, 1)  # (head_dim, query_len, key_len)
        relative_position_bias = relative_position_bias.unsqueeze(0).repeat(self.heads, 1, 1, 1)  # (heads, head_dim, query_len, key_len)
        relative_position_bias = relative_position_bias.mean(dim=1)  # (heads, query_len, key_len)
        return relative_position_bias

class MultiScaleAttention(nn.Module):
    def __init__(self, embed_size, heads, scales=[1, 3, 5], dropout=0.1):
        """
        Multi-scale attention mechanism
        :param embed_size
        :param heads
        :param scales
        :param dropout
        """
        super(MultiScaleAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"

        self.scale_convs = nn.ModuleList([
            nn.Conv1d(embed_size, embed_size, kernel_size=scale, padding=scale // 2)
            for scale in scales
        ])
        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, value, key, query):
        N, seq_len, embed_size = query.shape

        # Multi-scale convolutional feature extraction
        multi_scale_features = []
        for conv in self.scale_convs:
            # (N, embed_size, seq_len)
            conv_out = conv(query.transpose(1, 2))
            # (N, seq_len, embed_size)
            conv_out = conv_out.transpose(1, 2)
            multi_scale_features.append(conv_out)

        multi_scale_features = torch.stack(multi_scale_features, dim=2)  # (N, seq_len, num_scales, embed_size)
        multi_scale_features = multi_scale_features.mean(dim=2)  # (N, seq_len, embed_size)

        values = self.values(multi_scale_features).view(N, seq_len, self.heads, self.head_dim)
        keys = self.keys(multi_scale_features).view(N, seq_len, self.heads, self.head_dim)
        queries = self.queries(multi_scale_features).view(N, seq_len, self.heads, self.head_dim)

        attention_scores = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        out = torch.einsum("nhqk,nkhd->nqhd", [attention_weights, values])
        out = out.reshape(N, seq_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out

class AdaptiveAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout=0.1):
        """
        自适应注意力机制
        :param embed_size: 输入的特征维度
        :param heads: 多头注意力的头数
        :param dropout: Dropout 概率
        """
        super(AdaptiveAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(embed_size, heads),
            nn.Sigmoid()
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, value, key, query):
        N, seq_len, embed_size = query.shape

        values = self.values(value).view(N, seq_len, self.heads, self.head_dim)
        keys = self.keys(key).view(N, seq_len, self.heads, self.head_dim)
        queries = self.queries(query).view(N, seq_len, self.heads, self.head_dim)

        attention_scores = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        adaptive_weights = self.gate(query).view(N, seq_len, self.heads, 1)  # (N, seq_len, heads, 1)
        adaptive_weights = adaptive_weights.permute(0, 2, 1, 3)  # 转换为 (N, heads, seq_len, 1)
        adaptive_attention = attention_weights * adaptive_weights  # 动态加权
        adaptive_attention = self.dropout(adaptive_attention)

        out = torch.einsum("nhqk,nkhd->nqhd", [adaptive_attention, values])
        out = out.reshape(N, seq_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out


class DynamicConvolutionalAttention(nn.Module):
    def __init__(self, embed_size, heads, kernel_size=3, dropout=0.1):
        """
        Dynamic Convolutional Attention Mechanism
        :param embed_size
        :param heads
        :param kernel_size
        :param dropout
        """
        super(DynamicConvolutionalAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert self.head_dim * heads == embed_size, "Embedding size must be divisible by heads"

        self.values = nn.Linear(embed_size, embed_size, bias=False)
        self.keys = nn.Linear(embed_size, embed_size, bias=False)
        self.queries = nn.Linear(embed_size, embed_size, bias=False)
        self.fc_out = nn.Linear(embed_size, embed_size)
        self.conv = nn.Conv1d(self.head_dim * self.heads, self.head_dim * self.heads, kernel_size=kernel_size,
                              padding=kernel_size // 2, groups=self.heads)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, value, key, query):
        N, seq_len, embed_size = query.shape

        values = self.values(value).view(N, seq_len, self.heads, self.head_dim)
        keys = self.keys(key).view(N, seq_len, self.heads, self.head_dim)
        queries = self.queries(query).view(N, seq_len, self.heads, self.head_dim)

        attention_scores = torch.einsum("nqhd,nkhd->nhqk", [queries, keys]) / self.scale
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        conv_values = values.permute(0, 2, 3, 1).reshape(N, self.heads * self.head_dim, seq_len)
        conv_values = self.conv(conv_values).reshape(N, self.heads, self.head_dim, seq_len).permute(0, 3, 1, 2)

        out = torch.einsum("nhqk,nqhd->nqhd", [attention_weights, conv_values])
        out = out.reshape(N, seq_len, self.heads * self.head_dim)
        out = self.fc_out(out)
        return out


class MeSFormerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion, forwordatt_type='sq',atten_type='pool_att'):
        super(MeSFormerBlock, self).__init__()
        self.atten_type = atten_type
        if atten_type == 'self_att':
            self.attention = SelfAttention(embed_size=embed_size, heads=heads,)
            self.norm1 = nn.LayerNorm(embed_size)
        elif atten_type == 'ufo_att':
            self.attention = UFOAttention(embed_size=embed_size, heads=heads, dropout=dropout)
            self.norm1 = nn.LayerNorm(embed_size)
        elif atten_type == 'rpe_att':
            self.attention = RelativePositionEncodingAttention(embed_size=embed_size, heads=heads, dropout=dropout)
            self.norm1 = nn.LayerNorm(embed_size)
        elif atten_type == 'msa_att':
            self.attention = MultiScaleAttention(embed_size=embed_size, heads=heads, dropout=dropout)
            self.norm1 = nn.LayerNorm(embed_size)
        elif atten_type == 'Ad_att':
            self.attention = AdaptiveAttention(embed_size=embed_size, heads=heads, dropout=dropout)
            self.norm1 = nn.LayerNorm(embed_size)
        elif atten_type == 'Cro_att':
            self.attention = CrossAttention(embed_size=embed_size, heads=heads, dropout=dropout)
            self.norm1 = nn.LayerNorm(embed_size)
        elif atten_type == 'DC_att':
            self.attention = DynamicConvolutionalAttention(embed_size=embed_size, heads=heads, dropout=dropout)
            self.norm1 = nn.LayerNorm(embed_size)
        elif atten_type == 'Var_att':
            self.attention = VariationalAttention(embed_size=embed_size, heads=heads, dropout=dropout)
            self.norm1 = nn.LayerNorm(embed_size)
        elif atten_type == 'ufo_self_att':
            self.attention = UFOSelfAttention(embed_size=embed_size, heads=heads, dropout=dropout)
            self.norm1 = nn.LayerNorm(embed_size)
        elif atten_type == 'pool_att':
            self.token_mixer = Pooling()
            # self.drop_path = DropPath(dropout)
            # self.dropout = nn.Dropout(dropout)

            self.norm1 = LayerNormChannel(embed_size)

        self.forwordatt_type = forwordatt_type
        if self.forwordatt_type == 'sq':
            self.sqlayer = SQLayer1d(embed_size)
        elif self.forwordatt_type == 'original':
            self.norm2 = nn.LayerNorm(embed_size)
            self.feed_forward = nn.Sequential(
                nn.Linear(embed_size, forward_expansion * embed_size),
                nn.ReLU(),
                nn.Linear(forward_expansion * embed_size, embed_size)
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query):
        if self.atten_type == 'pool_att':
            # print(self.norm1(query).size(),'111111111')
            x = query + self.dropout(self.token_mixer(self.norm1(query)))
        else:
            attention = self.attention(value, key, query)
            x = self.dropout(self.norm1(attention + query))

        if self.forwordatt_type =='sq':
            x = self.sqlayer(x)
        elif self.forwordatt_type =='original':
            forward = self.feed_forward(x)
            x = self.dropout(self.norm2(forward + x))
            x = self.dropout(self.norm2(forward + x))

        return x


#
class MeSFormer(nn.Module):
    def __init__(self, embed_size=64, heads=4, num_layers=2, forward_expansion=4, dropout=0.3,
                 forwordatt_type='sq',atten_type='pool_att',device="cuda"):
        super(MeSFormer, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.trans_layers = nn.ModuleList(
            [MeSFormerBlock(
                    embed_size=embed_size,
                    heads=heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                    forwordatt_type = forwordatt_type,
                    atten_type=atten_type
                )for _ in range(num_layers)])

    def forward(self, gcn):
        b, c, n = gcn.size()
        # print(gcn.size(),'44444444444')

        for layer in self.trans_layers:
            # Query, Key and Value same in Encoder
            out = layer(gcn, gcn, gcn)
            # out = layer(gcn, gcn, gcn) + gcn
            out = out.reshape(b, c * n)

        return out





