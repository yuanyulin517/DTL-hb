#基于边进行计算的编码器，如果边密度不高，比如1000节点只有不到5000条边的情况则建议使用该编码器进行训练，只需要修改文件名即可切换。

import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
import numpy as np
from torch import nn
import math

from utils_MWVC.function import normalization

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 残差链接
class SkipConnection(nn.Module):

    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x , *input, **kwargs):
        return x + self.module(x ,*input, **kwargs)


class MultiHeadAttention(nn.Module):
    """
    使用 PyG 的 TransformerConv 实现的多头图注意力。
    适用于自注意力场景，输入节点特征 x 和边 e，
    输出经图注意力聚合后的节点特征。

    参数:
        n_heads (int): 注意力头数
        input_dim (int): 输入节点特征维度
        embed_dim (int, optional): 输出维度。若为 None，则必须提供 val_dim。
        val_dim (int, optional): 每个注意力头的值维度。若为 None，则从 embed_dim 计算：embed_dim // n_heads。
        key_dim (int, optional): 每个注意力头的键/查询维度。原始代码中默认为 val_dim，此处保留但未使用（TransformerConv 内部固定为 val_dim）。
    """
    def __init__(self, n_heads, input_dim, embed_dim, val_dim=None, key_dim=None):
        super().__init__()

        # 处理维度参数（与原始代码兼容）
        if val_dim is None:
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim  # 保留但未直接使用

        # TransformerConv 层：每个头输出 val_dim，拼接后维度为 n_heads * val_dim
        self.conv = TransformerConv(
            in_channels=input_dim,
            out_channels=val_dim,
            heads=n_heads,
            concat=True,          # 拼接多头输出
            beta=False,           # 不使用额外的 beta 参数，标准 Transformer 注意力
            dropout=0.0,          # 此处不设 dropout，与原始代码一致
            root_weight=False,    # 不额外添加自身特征的线性变换，聚合结果完全来自邻居
            bias=True
        )

        # 如果 embed_dim 不等于 n_heads * val_dim，添加一个线性层进行映射
        if self.embed_dim != n_heads * val_dim:
            self.out_proj = nn.Linear(n_heads * val_dim, self.embed_dim)
        else:
            self.out_proj = nn.Identity()

        # 初始化参数（可选，TransformerConv 内部已初始化，此处可省略）
        self.init_parameters()

    def init_parameters(self):
        # 与原始代码类似，对自定义参数（如果有）进行初始化
        for param in self.parameters():
            if param.dim() >= 2:  # 只对权重矩阵做均匀初始化
                stdv = 1. / math.sqrt(param.size(-1))
                param.data.uniform_(-stdv, stdv)

    def forward(self, x, e):
        """
        前向传播
        参数:
            q: 查询节点特征，形状 (batch_size, num_nodes, input_dim)
            e: 边, 元组(bat_index, i_idx, j_idx)
        返回:
            形状 (batch_size, num_nodes, embed_dim) 的节点特征
        """

        batch_size, num_nodes, _ = x.shape

        b_idx, i_idx, j_idx = e

        # 计算全局节点偏移：每个 batch 的节点索引偏移量为 batch_idx * num_nodes
        offset = b_idx * num_nodes
        src = j_idx + offset            # 源节点全局索引
        dst = i_idx + offset            # 目标节点全局索引
        edge_index = torch.stack([src, dst], dim=0)  # (2, num_edges)

        # 2. 将所有节点特征拼接成一维
        x_flat = x.contiguous().view(batch_size * num_nodes, self.input_dim)  # (total_nodes, input_dim)

        # 3. 通过 TransformerConv 计算注意力聚合
        out_flat = self.conv(x_flat, edge_index)  # (total_nodes, n_heads * val_dim)

        # 4. 投影到 embed_dim（如果需要）
        out_flat = self.out_proj(out_flat)        # (total_nodes, embed_dim)

        # 5. 恢复批处理形状
        out = out_flat.view(batch_size, num_nodes, self.embed_dim)

        return out

# BatchNorm
# 归一化
class Normalization(nn.Module):
    '''
        一个可选择的归一化模型，根据normalization参数选择
    '''
    def __init__(self, embed_dim, normalization='batch'):
        super().__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # 默认情况下，归一化使用偏置0和权重unif(0,1)初始化仿射参数，这太大了
        self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input):
        #根据模型选择运行方式
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "未知的归一化类型"
            return input


# 论文中提到的AttentionLayer
class MultiHeadAttentionLayer(nn.Module):
    '''
        单个注意力层,GraphAttentionEncoder的组件之一，具体流程建议参考论文的编码器部分
        构造参数：
            n_heads:头数
            embed_dim:输入及输出维度
            feed_forward_hidden:传到下一层之前的前馈神经网络的隐藏层维度数，<=0时默认不设置隐藏层
            normalization:选择的归一化模型
    '''
    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super().__init__()
        self.MHA = SkipConnection(  # 论文中提到的MHA子层
            MultiHeadAttention(
                n_heads,
                input_dim=embed_dim,
                embed_dim=embed_dim
            )
        )
        self.norm = Normalization(embed_dim, normalization)
        self.out = SkipConnection(  # 前馈子层
            nn.Sequential(
                # 输入映射层
                nn.Linear(embed_dim, feed_forward_hidden),
                # 用来增加非线性性，使得神经网络能够拟合更复杂的函数
                # 激活层
                nn.ReLU(),
                # 输出映射层
                nn.Linear(feed_forward_hidden, embed_dim)
            ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim)
        )
        self.norm_2 = Normalization(embed_dim, normalization)

    def forward(self,  *input, **kwargs):
        h=self.MHA( *input, **kwargs)
        return self.norm_2(self.out(self.norm(h)))

class GraphAttentionNetwork(nn.Module):
    def __init__(self, n_heads,
                 embed_dim,
                 n_layers=1,
                 feed_forward_hidden=512,
                 normalization='batch'):
        super().__init__()
        # 创建了一个由n_layers个MultiHeadAttentionLayer实例组成的序列
        # 每个实例都使用相同的参数。他们作为Sequential实例的参数
        self.layers = nn.ModuleList([
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ])

    def forward(self, input , e):
        for layer in self.layers:
            input = layer(input, e)
        return input
        
class GraphAttentionEncoder(nn.Module):
    def __init__(self, n_heads,
                 embed_dim,
                 node_dim,
                 n_layers=2,
                 feed_forward_hidden=512,
                 normalization='batch'):
        super().__init__()
        # 将输入映射到嵌入空间
        self.init_embed = nn.Linear(node_dim, embed_dim)

        self.GAN_Q=GraphAttentionNetwork(n_heads,embed_dim, n_layers, feed_forward_hidden, normalization)

    def forward(self, dataset):
        e=dataset['e']
        input=get_input(dataset)
        input = self.init_embed(input)
        batch_size,n_size,_=input.size()
        h_q=self.GAN_Q(input, e)
        return h_q


def get_input(dataset):
    mat = dataset['graphs_matrix']      #邻接矩阵
    w = dataset['w']                    #结点权重
    degree = mat.sum(-1).float()        #度数
    dims = []                             #保存特征的列表
    # 第一个特征维度：权重
    # dims.append(w/w.max(-1)[0].unsqueeze(-1))
    dims.append(normalization(w))
    # 第二个特征维度：度数
    dims.append(normalization(degree))
    # 第三个特征维度：度数/权重
    dims.append(normalization(degree / w))
    # 第四个特征维度：邻居权重和
    dims.append(normalization((mat * w.unsqueeze(1)).sum(-1).float()))

    dims=torch.stack(dims,dim=-1)
    return dims
