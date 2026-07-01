
import torch.nn as nn
import torch
from utils_MWVC.function import normalization
from utils_MWVC.profiling import NULL_TIMER
class multi_Linear(nn.Module):
    def __init__(self,
                 input_dim,
                 out_dim,
                 feed_forward_hidden=None,
                 hidden_layers=2):
        super().__init__()
        if feed_forward_hidden is None or feed_forward_hidden <=0:
            feed_forward_hidden=input_dim*2
        self.forward_linear = nn.Sequential(
            nn.Linear(input_dim, feed_forward_hidden),
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(feed_forward_hidden, feed_forward_hidden),
            nn.ReLU()) for _ in range(hidden_layers)],
            nn.Linear(feed_forward_hidden,input_dim),
            nn.Linear(input_dim,out_dim))

    def forward(self,x):
        return self.forward_linear(x)

class actor_model(nn.Module):
    """
    基于图的 Actor 网络，用于在强化学习中选择动作。
    输入节点特征和图掩码，输出每个节点的动作概率分布（非法节点概率为0）。
    """
    def __init__(self,
                 input_dim,                     # 最终输入到线性层的特征维度（应为节点特征维度的两倍）
                 feed_forward_hidden=None,      # 隐藏层维度列表（若为None，可能使用默认值）
                 hidden_layers=2):              # 隐藏层数量
        super().__init__()
        self.input_dim = input_dim

        # 构建多层全连接网络，将特征映射到标量分数（输出维度为1）
        self.linears = multi_Linear(input_dim = input_dim *2, 
                                    out_dim=1,
                                    feed_forward_hidden = feed_forward_hidden,
                                    hidden_layers=hidden_layers)

    def forward(self, encoder, dataset, ans, mask_out, age, index_batch = None, timer=None, tag=''):
        """
        前向传播
        :param encoder: 节点特征，形状 (batch_size, node_size, dim)
        :param dataset: 图的所有数据信息
        :param mask_graph: 图聚合掩码，形状 (batch_size, node_size)，用于指示哪些节点参与全局图特征计算
        :param mask_out: 动作掩码，形状 (batch_size, node_size)，指示哪些节点是合法动作
        :param index_batch: 样本掩码，形状 (batch_size)，指示哪些样本需要计算
        :return: 动作概率分布，形状 (batch_size, node_size)，合法节点概率之和为1，非法节点概率为0
        """
        if timer is None:
            timer = NULL_TIMER
        w = dataset['w']
        if index_batch is not None:
            encoder = encoder[index_batch]
            ans = ans[index_batch]
            mask_out = mask_out[index_batch]
            w = w[index_batch]

        
        # 求取动态特征
        with timer.seg('move_emd'):
            move_emd = get_move_emd(dataset, age, ans, index_batch = index_batch)
        with timer.seg('move_emd_assert'):
            assert (move_emd == move_emd).all(), "move_emd不应包含任何 NaN"
        with timer.seg('actor_fwd_' + tag):
            #与动态特征进行合并
            encoder = torch.cat([encoder, move_emd], dim = -1)

            batch_size, node_size, dim = encoder.shape
            # 验证输入维度：节点特征维度 dim 必须等于 self.input_dim
            # assert dim == self.input_dim, f"维度不匹配，需要{self.input_dim}，但提供了{dim}"

            # 计算全局图特征
            x = (ans*w).unsqueeze(-1)
            graph = (encoder * x).sum(1)/ x.sum(1)

            # 将动作掩码展平为一维，便于后续索引
            mask_out = mask_out.view(-1)

            # 将节点特征与全局图特征拼接：先复制 graph 到每个节点，然后在最后一维拼接，得到 (batch_size, node_size, 2*dim)
            # 再重塑为 (batch_size * node_size, input_dim)，每一行对应一个节点的完整特征
            encoder = torch.cat([encoder, graph.unsqueeze(1).expand(batch_size, node_size, dim)], dim=-1)
            encoder = encoder.view(-1, self.input_dim *2)
            # 获取所有合法节点的索引（mask_out 中为 True 的位置）
            true_index = torch.where(mask_out)[0]

            # 仅对合法节点应用线性网络，计算动作分数，输出形状 (batch_size * node_size, 1)
            out = self.linears(encoder[true_index])
            # out_w = self.linears_w(encoder_w.contiguous().view(-1, self.input_dim)[true_index])
            # 初始化一个全 -inf 的向量，长度等于所有节点总数（展平后）
            ans = torch.full((batch_size * node_size,), float('-inf'), device=encoder.device)
            # 将合法节点的分数填入对应位置，非法节点保持 -inf
            ans[true_index] = out.squeeze()
            # 重塑为 (batch_size, node_size)，并在节点维度上做 softmax，非法节点因 -inf 概率为0
            ans = torch.softmax(ans.view(batch_size, node_size), dim=-1)

        return ans

class critic_model(nn.Module):
    def __init__(self,
                 input_dim,
                 feed_forward_hidden=None,
                 hidden_layers=2):
        super().__init__()
        self.input_dim = input_dim
        self.linears=multi_Linear(input_dim=input_dim , 
                                  out_dim=1, 
                                  feed_forward_hidden=feed_forward_hidden, 
                                  hidden_layers=hidden_layers)

    def forward(self, encoder, dataset, ans, age):
        move_emd = get_move_emd(dataset, age, ans)
        encoder = torch.cat([encoder, move_emd], dim=-1)
        batch_size,node_size,dim=encoder.shape
        assert dim==self.input_dim,"维度不匹配，需要{}，但提供了{}".format(self.input_dim , dim)
        # 计算全局图特征
        x = (ans*dataset['w']).unsqueeze(-1)
        
        graph = (encoder * x).sum(1)/ x.sum(1)
        
        return self.linears(graph).squeeze()

def get_move_emd(dataset, age, ans, index_batch = None) :
    mat = dataset['graphs_matrix']                            #邻接矩阵
    w = dataset['w_norm']
    if index_batch is not None:
        mat = mat[index_batch]
        age = age[index_batch]
        w = w[index_batch]
    
    degree = mat.sum(-1).float()        #度数

    dims = []                            #保存特征的列表
    # 第一个特征维度：邻居覆盖率
    dims.append((mat * ans.unsqueeze(-2)).sum(-1)/(degree + 1e-8))
    # # 第二个特征维度：age
    dims.append(normalization(age))
    # 第三个特征维度：uncover_degree
    uncover_degree = (mat * (~ ans).unsqueeze(-2)).sum(-1).float()
    dims.append(normalization(uncover_degree / w))
    
    dims=torch.stack(dims,dim=-1)
    return dims

