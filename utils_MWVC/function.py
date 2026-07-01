import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from torch.utils.data import Dataset
from ConstructWVC import ConstructWVC as CWVC
from utils_MWVC.profiling import NULL_TIMER
from utils_MWVC.weighting import sample_weights_torch

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def move_to(var, device):
    if isinstance(var, dict):
        return {k: move_to(v, device) for k, v in var.items()}
    return var.to(device)


def ConstructWVC(dataset):
    #除了进行初始解集计算外，还进行了各数据的初始化
    mat=dataset['graphs_matrix']
    dataset['e'] = torch.where(mat)
    w=dataset['w']
    ans=[]
    ans_w=[]
    e_bat, e_x, e_y = dataset['e']
    w_numpy = w.cpu().numpy()
    for i in range(w.size(0)):
        sol = CWVC(w_numpy[i], e_x[e_bat == i].cpu().numpy(), e_y[e_bat == i].cpu().numpy())
        sol.ConstructVC()
        ans.append(sol.get_ans())
        ans_w.append(sol.get_ans_w())
    dataset['ans']=torch.tensor(ans,dtype=torch.bool,device=device)
    dataset['bast_ans']=torch.tensor(ans,dtype=torch.bool,device=device)
    dataset['ans_w']=torch.tensor(ans_w,dtype=torch.float32,device=device)
    dataset['bast_ans_w']=torch.tensor(ans_w,dtype=torch.float32,device=device)
    dataset['taboo']=torch.zeros(dataset['ans'].size(),device=device)
    dataset['conf']=torch.ones(dataset['ans'].size(),device=device).bool()
    dataset['age']=torch.zeros(dataset['ans'].size(),device=device)
    dataset['w_norm'] = w/w.std(dim = -1, unbiased=False).unsqueeze(-1)
    dataset['node_all'] = (mat.sum(-1)>0)          #非孤立顶点

def validate_restart(models, dataset, opts, is_print=False, restart_num =8, batch= 4):
    ans_w = None
    start_time = time.time()
    for i in range(restart_num) :
        dataset_restart = data_restart(dataset, batch)
        ans_w_i = validate(models, dataset_restart, opts)
        ans_w_i = ans_w_i.view(batch, -1).min(0)[0]
        if ans_w is None :
            ans_w = ans_w_i
        else :
            mask = ans_w > ans_w_i
            ans_w[mask] = ans_w_i[mask]
        dataset_restart = None

        if is_print :
            print('当前解平均：{}，最优解平均：{}'.format(ans_w_i.mean(), ans_w.mean()))
    end_time = time.time()
    return ans_w, end_time - start_time

def data_restart(dataset, num) :
    dataset_restart = {}
    for item in dataset :
        x = dataset[item]
        dataset_restart[item] = x.repeat(*((num,) + (1,) * (x.dim()-1)))
    return dataset_restart

def validate(models, dataset, opts, is_print=False, timer=None, deterministic=False):
    """
    评估函数：在给定数据集上运行多次优化步骤，更新解并记录最优解。
    models: 包含编码器、删除演员、添加演员的模型字典
    dataset: 包含图数据、权重、当前解、禁忌表等信息的字典
    opts: 配置参数，包含评估步数、日志步长、禁忌步数等
    is_print: 是否打印中间日志
    timer: CudaSegTimer，传入则做分段计时；None 时行为与原版完全一致
    deterministic: True 时验证完全确定性——动作选择用 argmax 贪心、冗余删除去掉随机平局扰动，
                   使 checkpoint 选择指标无 rollout 随机噪声；False（默认）保持原采样行为，
                   供 validate_restart/eval_only 的多重启评估使用。训练路径(train_step)不经此函数。
    """
    if timer is None:
        timer = NULL_TIMER
    # 验证时选用确定性贪心或随机采样；不影响 train_step 的训练采样
    select_type = 'greedy' if deterministic else 'sampling'
    start_time = time.time()
    timer.begin()                              # 同步 + 启动总计时（禁用时 no-op）

    # 初始化数据集：构建图结构、初始化解等（CPU 为主，单独计时）
    timer.construct_start()
    if deterministic:
        # 验证(checkpoint 选择)路径：固定构造随机性，消除初始解逐-epoch 漂移。
        # seed=9100 与验证集一致；构造完立即恢复 random 状态，绝不打乱训练侧随机流。
        _rng_state = random.getstate()
        random.seed(9100)
        ConstructWVC(dataset)
        random.setstate(_rng_state)
    else:
        ConstructWVC(dataset)
    timer.construct_end()

    # 从数据集中取出常用张量
    graphs = dataset['graphs_matrix']          # 邻接矩阵，形状 (batch_size, node_size, node_size)
    conf = dataset['conf']                     # 配置变化
    age = dataset['age']                        # 修改时间
    w = dataset['w']                            # 节点权重，形状 (batch_size, node_size)
    ans = dataset['ans']                         # 当前解（布尔张量，True 表示节点在顶点覆盖中）
    batch_size, node_size = w.size()
    if is_print:
        print('平均初始解权重:{}'.format(dataset['ans_w'].mean()))

    # 编码器计算节点嵌入（只计算一次，因为使用的都是静态特征）
    with timer.seg('encode'):
        emd = models['encoder'](dataset)

    for step in range(opts.eval_num):               # 迭代指定的评估次数
        timer.set_active(step > 0)                  # 丢弃第一步（cudnn/首次调用初始化）
        # 每隔 log_step 步打印当前历史最优解的平均权重
        # if (step + 1) % opts.log_step == 0 and is_print:
        #     print("第{}步时的平均最优解权重:{}".format(step + 1, dataset['bast_ans_w'].mean()))

        # 构造可删除节点掩码：当前解中且不在禁忌表中的节点
        with timer.seg('search'):
            mask = dataset['ans'] & (dataset['taboo'] <= 0)

        # 删除演员网络输出删除概率，按概率采样第一个要删除的节点
        with timer.seg('decode'):
            p = models['delete_actor'](emd, dataset, ans, mask, age, timer=timer, tag='delete')
            delete_1 = select_node(p, mask, select_type, timer=timer)

        # 从解中移除该节点，并更新掩码/配置/时间
        with timer.seg('search'):
            zero_to_batch = torch.arange(batch_size, device=device)  # batch 索引 [0,1,...,batch_size-1]
            ans[zero_to_batch, delete_1] = False
            mask[zero_to_batch, delete_1] = False
            conf = conf | graphs[zero_to_batch, delete_1, :]
            conf[zero_to_batch, delete_1] = False
            # 注：此处不更新 delete_1 的 age，与 train_step 第一次删除保持一致（train 仅在第二次删除起更新 age）

        # 再次计算删除概率，选择第二个要删除的节点
        with timer.seg('decode'):
            p = models['delete_actor'](emd, dataset, ans, mask, age, timer=timer, tag='delete')
            delete_2 = select_node(p, mask, select_type, timer=timer)

        with timer.seg('search'):
            ans[zero_to_batch, delete_2] = False
            conf = conf | graphs[zero_to_batch, delete_2, :]
            conf[zero_to_batch, delete_2] = False
            age[zero_to_batch, delete_2] = step
            # 获取未覆盖的边对应的节点（即需要添加的节点）
            add_node = get_no_cov(graphs, ans, conf)

        # 循环添加节点，直到所有边都被覆盖
        while add_node.any():
            with timer.seg('search'):
                mask = add_node                                     # 当前可添加节点掩码
                index_batch = torch.where(add_node.any(dim=-1))[0]  # 需要添加节点的 batch 索引
            with timer.seg('decode'):
                p = models['add_actor'](emd, dataset, ans, mask, age, index_batch, timer=timer, tag='add')
                add_select = select_node(p, mask[index_batch], select_type, timer=timer)
            with timer.seg('search'):
                ans[index_batch, add_select] = True                 # 将节点加入解
                conf[index_batch] = conf[index_batch] | graphs[index_batch, add_select, :]
                age[index_batch, add_select] = step
                add_node = get_no_cov(graphs, ans, conf)            # 重新计算未覆盖节点

        # 冗余删除 + 可行性校验 + 禁忌表/最优解更新
        with timer.seg('search'):
            ok_delete = get_ok_delete(graphs, ans)                  # 可安全删除的节点掩码
            while ok_delete.any():
                index_batch = torch.where(ok_delete.any(dim=-1))[0]
                # 选一个冗余点删除：deterministic 时取首索引(可复现)，否则随机扰动打破平局
                # 注：此处仅做确定化，不改删除准则；4.2 的"权重优先冗余删除"是另一件事，勿混
                ok_delete_int = ok_delete[index_batch].int()
                if deterministic:
                    delete_select = ok_delete_int.argmax(dim=-1)
                else:
                    delete_select = (ok_delete_int + torch.rand(ok_delete_int.size(), device=device) * 0.1).argmax(dim=-1)
                ans[index_batch, delete_select] = False
                ok_delete = get_ok_delete(graphs, ans)

            # 断言当前解是可行解（覆盖所有边）
            assert check_MVC(graphs, ans), '该解集不是可行解'

            # 更新禁忌表：新加入的节点（当前解中有而初始解中没有）设置禁忌步数
            taboo = ans & (~dataset['ans'])
            dataset['taboo'] = dataset['taboo'] - 1
            dataset['taboo'][taboo] = opts.taboo_num
            dataset['ans'] = ans

            # 计算当前解的总权重，并更新历史最优解
            w_temp = (ans * w).sum(dim=-1)
            dataset['ans_w'] = w_temp
            mask = w_temp < dataset['bast_ans_w']
            dataset['bast_ans'][mask] = ans[mask]
            dataset['bast_ans_w'][mask] = w_temp[mask]

        timer.step_done()

    timer.end()                                # flush + 同步 + 停止总计时

    # 打印总耗时和最终平均最优解权重
    if is_print :
        print("花费时间：{} \n最终平均解集大小为：{}".format(
            time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)),
            dataset['bast_ans_w'].mean()))

    return dataset['bast_ans_w']                                     # 返回历史最优权重（供记录或绘图）

def create_data_tensor(dataset, opts, is_random=True):
    size = opts.text_node_size
    for data_item in dataset:
        size = max(size, data_item['v_size'])
        
    device = opts.device
    # 使用numpy进行预处理（通常比Python循环快）
    data_list = []
    w_list=[]
    ans_list = []
    ans_w_list=[]
    for data_item in dataset:
        n=data_item['v_size']
        # 生成随机索引
        if is_random:
            indices = np.random.permutation(size)
        else:
            indices = np.arange(size)

        # 创建邻接矩阵
        adj_matrix = np.zeros((size, size), dtype=np.float32)
        edges = np.array(data_item['data'])
        if len(edges) > 0:
            # 应用索引映射
            mapped_edges = indices[edges]
            u = mapped_edges[:, 0]
            v = mapped_edges[:, 1]
            adj_matrix[u, v] = 1
            adj_matrix[v, u] = 1
        #创建权重矩阵
        w=np.ones(size,dtype=np.float32)
        w[indices[:n]] = np.array(data_item['w'],dtype=np.float32)
        # 创建答案向量
        ans_vector = np.zeros(size, dtype=np.int32)
        ans_w=0.0
        if 'ans' in data_item:
            mapped_ans = indices[np.array(data_item['ans'])]
            ans_vector[mapped_ans] = 1
            ans_w=data_item['ans_w']

        data_list.append(adj_matrix)
        w_list.append(w)
        ans_list.append(ans_vector)
        ans_w_list.append(ans_w)

    # 批量转换为张量
    data_tensor = torch.tensor(np.stack(data_list), device=device).bool()
    w_tensor=torch.tensor(np.stack(w_list), device=device)
    ans_tensor = torch.tensor(np.stack(ans_list), device=device).bool()
    ans_w_tensor = torch.tensor(np.stack(ans_w_list), device=device)

    return data_tensor,w_tensor, ans_tensor,ans_w_tensor

def generate_graphs(n_size, min_p, max_p, count):
    dataset = {}
    graphs_matrix=torch.rand((count, n_size, n_size), dtype=torch.float32, device=device)
    p=torch.rand(count, device=device)*(max_p-min_p)+min_p
    graphs_matrix=graphs_matrix < p.unsqueeze(-1).unsqueeze(-1)
    mask=~torch.triu(torch.ones((n_size, n_size), device=device)).bool()
    graphs_matrix=graphs_matrix&mask.unsqueeze(0)
    w = sample_weights_torch((count, n_size), device)
    graphs_matrix = (graphs_matrix | graphs_matrix.transpose(1, 2))
    e_size = graphs_matrix.sum(-1).sum(-1) / 2  # 对称矩阵每边计两次，除2
    v_size = (graphs_matrix.sum(-1) > 0).sum(-1)  # 此时按行求和=真实度数
    dataset['e_size']=e_size
    dataset['v_size']=v_size
    dataset['w']=w
    dataset['graphs_matrix']=graphs_matrix
    dataset['e'] = torch.where(graphs_matrix)
    dataset['ans']=torch.zeros((count, n_size), device=device).bool()
    dataset['ans_w']=torch.zeros(count,dtype=torch.float32,device=device)
    dataset['ans_size']=torch.zeros(count,device=device)
    # dataset['count']=count
    return dataset


def check_MVC(x,ans):
    return ~(x.bool()&(~ans.unsqueeze(1))&(~ans.unsqueeze(2))).any()

def dict_marge(*dicts):
    dataset={}
    for item in dicts[0]:
        if torch.is_tensor(dicts[0][item]):
            dataset[item]=torch.cat([x[item] for x in dicts],dim=0)
        elif isinstance(dicts[0][item], (int, float, complex)):
            dataset[item]=0
            for x in dicts:
                dataset[item]+=x[item]
    return dataset

def load_dataset_to_tensor(load_dataset, opts, is_random=True):
    count=len(load_dataset)
    graphs_matrix,w,ans,ans_w=create_data_tensor(load_dataset,opts, is_random)
    dataset={}
    dataset['graphs_matrix']=graphs_matrix
    dataset['w']=w
    dataset['ans']=ans
    dataset['ans_w']=ans_w
    # dataset['count']=count
    dataset['v_size']=(graphs_matrix.sum(-1)>0).sum(-1)
    dataset['e_size']=graphs_matrix.sum(-1).sum(-1)/2
    dataset['ans_size']=ans.sum(-1)
    return dataset

def models_eval(models):
    for i in models:
        models[i].eval()

def models_train(models):
    for i in models:
        models[i].train()

def update_models_to_eval(models):
    for str in models:
        models[str].eval()


def select_node(p, mask,type, timer=None):
    if timer is None:
        timer = NULL_TIMER
    with timer.seg('assert_sync'):
        assert (p == p).all(), "概率不应包含任何 NaN"

    if type == "greedy":
        _, selected = p.max(1)
        assert not (~mask).gather(1, selected.unsqueeze(-1)).data.any(), "贪心解码错误: infeasible action has maximum probability"  # 不能重复访问访问节点
        # 贪心解码是一种简单的解码策略，它在每一步决策时，都会选择当前状态下看起来最优的那个选项，而不考虑这个选择对后续步骤的影响。
    elif type == "sampling":  # 采样编码策略, 即使对同样的输入，采样每次获得的数据都不同
        """
            为什么强化学习一定要用sampling？
                1. 动作选择的随机性需求
                    在强化学习中，智能体需要根据策略网络输出的动作概率分布来选择动作。
                    这种选择不能总是确定性的，因为如果智能体总是选择概率最高的动作，就会导致探索不足。
                    例如，在一个迷宫探索游戏中，如果智能体总是朝着它认为最有可能获得奖励的方向移动，可能会错过其他潜在的、更优的路径。
                    torch.multinomial函数可以根据给定的动作概率分布进行随机抽样，使得智能体能够以一定的概率选择不同的动作，从而实现探索与利用的平衡。
                    假设策略网络输出了动作概率分布action_probs = [0.1, 0.3, 0.6]，这表示有 3 个动作，它们被选中的概率分别为 0.1、0.3 和 0.6。
                    用torch.multinomial(action_probs, 1)（假设只选择一个动作），
                    智能体就可以以这些概率随机地选择一个动作，有时候会选择概率较低的动作，这就增加了探索环境的机会。
                2. 符合概率分布的动作选择
                    强化学习的策略本质上是一个概率分布，它定义了在每个状态下选择每个动作的概率。
                    torch.multinomial能够准确地从这个概率分布中进行抽样，保证动作选择的概率符合策略网络所学习到的分布。
                    这对于正确地训练策略网络非常重要，因为训练过程依赖于智能体按照策略所规定的概率进行动作选择，然后根据反馈来更新策略。
                    例如，在基于策略梯度的算法中，如 A2C 或 A3C 算法，
                    策略网络输出动作概率，智能体使用torch.multinomial选择动作并与环境交互，得到奖励。
                    然后根据奖励和动作概率来计算策略梯度，以更新策略网络，使得策略网络输出的概率分布能够朝着获得更多奖励的方向调整。
                    如果动作选择不按照正确的概率分布进行，策略梯度的计算就会出现偏差，导致训练效果不佳。
                """

        with timer.seg('sample'):
            selected = p.multinomial(1).squeeze(1)
            while (~mask).gather(1, selected.unsqueeze(-1)).data.any():
                print('采样到错误值，重新采样!')
                selected = p.multinomial(1).squeeze(1)

    else:
        assert False, "未知解码类型"
    return selected

def batch_check_MVC(x, ans):
    """
    批量检查给定的解集 ans 是否为图的顶点覆盖。
    """
    return ~(x.bool() & (~ans.unsqueeze(1)) & (~ans.unsqueeze(2))).any(dim=(1,2))


def get_no_cov(x, ans, conf):
    """
    找出所有存在'未覆盖边'的顶点
    """
    return (x.bool() & (~ans.unsqueeze(1))).any(dim=-1) & (~ans) & conf


def get_ok_delete(x, ans):
    """
    找出当前顶点覆盖中可以安全删除的顶点（即删除后仍能保持覆盖性质）。
    """
    return (~((x.bool() & ~ans.unsqueeze(1)).any(dim=2))) & ans

def get_models_state_dict(models):
    ans={}
    ans['encoder']=models['encoder'].state_dict()
    ans['add_actor']=models['add_actor'].state_dict()
    ans['delete_actor']=models['delete_actor'].state_dict()
    ans['critic']=models['critic'].state_dict()
    return ans


def normalization(x, dim=-1):
    return x / (x.max(dim=dim)[0].unsqueeze(dim) + 1e-8)
    