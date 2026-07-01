import torch
import torch.nn.functional as F
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


# 多头注意力
class MultiHeadAttention(nn.Module):
    '''
        一个普通的多头注意力模型实现，MultiHeadAttentionLayer的组件之一
        构造参数：
            n_heads:头数
            input_dim:输入维度数
            embed_dim:输出给下一层的维度数，
            val_dim:v的维度数，
            key_dim:q和k的维度数
        embed_dim这个参数为None的话，运行时估计会报错(未验证),所以构造时一般会有这个参数
        embed_dim可被n_heads整除，此处未检查，但AttentionModel类调用时检查了
    '''

    def __init__(
            self,
            n_heads,  # 头数
            input_dim,  # 输入维度
            embed_dim=None,  # 嵌入维度
            val_dim=None,  # v的维度
            key_dim=None  # q和k的维度
    ):
        super().__init__()

        if val_dim is None:
            assert embed_dim is not None, "提供 embed_dim 或 val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # 参见 Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))
        # 初始化参数
        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            # 计算一个标准差值，让参数的初始值满足均匀分布，避免梯度爆炸或者梯度消失
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k=None, v=None, mask=None):
        """
        参数：
            q: 查询向量,形状(批数, 查询结点数, 输入维度)
            k: 键向量,形状(批数, 图的结点数, 输入维度)
            v: 数据向量,形状(批数, 图的结点数, 输入维度)
            mask: 掩码 ，形状(批数, 查询结点数, 图结点数)或其他可广播的结点形状
        mask中,1表示需要屏蔽（不参与注意力计算）,0表示可以参与注意力计算
        k和v为空时,代表进行自注意力计算,k和v赋值为q
        输出:
        """
        if k is None:
            k = q  # 计算自注意力
        if v is None:
            v = q

        # 获取批数,图的结点数,维度数,查询结点数,h 应该是 (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = k.size()
        n_query = q.size(1)
        # 检查输入数据的形状是否有问题
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim, "Wrong embedding dimension of query:{},{},{}".format(q.size(), k.size(),
                                                                                            input_dim)
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        vflat = v.contiguous().view(-1, input_dim)
        kflat = k.contiguous().view(-1, input_dim)  # 改变形状为(batch_size*graph_size, input_dim)
        qflat = q.contiguous().view(-1, input_dim)  # 改变形状为(batch_size*n_query, input_dim)

        # 定义计算时的形状 最后一个维度对于键和值可以不同
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # 计算输入数据和查询得出Q,K,V，并按结点维度拆分为n_heads头
        Q = torch.matmul(qflat, self.W_query).view(shp_q)  # 形状(n_heads, n_query, n_query, dim/n_heads)
        K = torch.matmul(kflat, self.W_key).view(shp)  # 形状(n_heads, batch_size, graph_size, dim/n_heads)
        V = torch.matmul(vflat, self.W_val).view(shp)  # 形状(n_heads, batch_size, graph_size, dim/n_heads)

        # 计算注意力分数,即Q*K^T/sqrt(d_k)这一步
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2,
                                                                       3))  # 形状 (n_heads, batch_size, n_query, graph_size)
        # 可选地应用掩码以防止注意力
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask] = -1e9
        # attn=softmax(Q*K^T/sqrt(d_k))
        attn = F.softmax(compatibility, dim=-1)

        # 如果有节点没有邻居，则softmax返回nan，因此我们将它们固定为0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        # heads=softmax(Q*K^T/sqrt(d_k))*V
        heads = torch.matmul(attn, V)  # 形状 (n_heads, batch_size, n_query, val_dim)

        # 将每个头连接起来，并经过最后一层线性层还原成输出维度，最后改变形状为(batch_size, n_query, embed_dim)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

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

    def forward(self, input , mask=None):
        for layer in self.layers:
            input = layer(input, mask=mask)
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
        h_q=self.GAN_Q(input, ~dataset['graphs_matrix'])
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
