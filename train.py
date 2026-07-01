import copy
import os
import time
import torch

import utils_MWVC.function as F

from utils_MWVC.function import validate

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

flag = 0
critic_losses=[]
actor_losses=[]

def train_epoch(models,dataset,optim,opts,epoch,lr_scheduler):

    #以下三个变量是用来记录数据的
    global flag
    global critic_losses
    global actor_losses

    start_time = time.time()
    train_dataset=F.generate_graphs(opts.n_size,opts.min_p,opts.max_p,opts.graph_size)
    F.ConstructWVC(train_dataset)
    print('epoch:{}  平均初始解权重：{}'.format(epoch, train_dataset['ans_w'].mean()))
    F.models_train(models)
    
    reward = []
    temp = {'p': [], 'ans': [], 'index': [], 'age': []} 
    p_new = []
    p_old = []
    actor_idx = []
    l = 0
    emd =  models['encoder'](train_dataset)
    for i in range(opts.step_num):
        flag += 1
        if (i+1)%opts.log_step==0 :
            print("第{}步的平均最优集权重为,{}".format(i+1, train_dataset['bast_ans_w'].mean()))
        if i%opts.updata_old==0 :
            models_old=copy.deepcopy(models)
            F.update_models_to_eval(models_old)
            emd_old=models_old['encoder'](train_dataset)
            
        reward_i, temp_i, p_new_i, p_old_i= train_step(models, models_old, emd, emd_old, train_dataset, optim, i+1, opts)
        
        reward += reward_i
        for s in ['p', 'ans', 'index', 'age'] :
            temp[s] += temp_i[s]
        p_new += p_new_i
        p_old += p_old_i
        actor_idx += [j for j in range(l, l+len(p_new_i))]
        l += len(reward_i)

        if (i+1)% opts.ppo_num ==0:
            # ---------- 计算PPO损失 ----------
            loss = compute_ppo_loss(emd, models['critic'],
                                    emd_old, models_old['critic'],
                                    train_dataset, reward, temp['p'], temp['ans'],
                                    torch.cat(p_new, dim=0),   # 将所有步的概率拼接成一维张量
                                    torch.cat(p_old, dim=0),
                                    temp['index'], temp['age'], actor_idx, opts)
    
            # ---------- 梯度下降更新模型参数 ----------
            optim.zero_grad()
            loss.backward()
            # 梯度裁剪
            for group in optim.param_groups:
                torch.nn.utils.clip_grad_norm_(group["params"], opts.max_grad_norm)
            optim.step()
            
            reward = []
            temp = {'p': [], 'ans': [], 'index': [], 'age': []} 
            p_new = []
            p_old = []
            actor_idx = []
            l = 0
            emd =  models['encoder'](train_dataset)
        
    lr_scheduler.step()
    # 验证平均回报有效性
    print('对测试数据集进行评估')
    F.models_eval(models)
    with torch.no_grad():
        validate(models, dataset, opts, True, deterministic=True)
    if (opts.checkpoint_epochs != 0 and epoch % opts.checkpoint_epochs == 0) or epoch == opts.n_epochs - 1:
        print('保存模型及其状态...')
        torch.save(
            {
                'models': F.get_models_state_dict(models),
                'optim': optim.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(opts.save_dir, 'epoch-{}.pt'.format(epoch))
        )
        
    dataset['all_critic_loss'].extend(critic_losses)
    dataset['all_actor_loss'].extend(actor_losses)
    dataset['all_ans_w'].append(dataset['bast_ans_w'].mean().item())
    critic_losses=[]
    actor_losses=[]
    epoch_duration = time.time() - start_time
    print("第 {} 轮耗时 {}\n".format(epoch + 1, time.strftime('%H:%M:%S', time.gmtime(epoch_duration))))

def train_step(models, models_old, emd, emd_old, train_dataset, optim, step, opts):
    """
    执行一步训练：从当前策略采样一条轨迹，收集经验，计算PPO损失并更新模型参数。
    """
    graphs = train_dataset['graphs_matrix']          # 邻接矩阵，形状 (batch, node, node)
    conf = train_dataset['conf']                     # 配置检查
    age = train_dataset['age']                       # 修改时间
    w = train_dataset['w']                           # 节点权重，形状 (batch, node)
    w_norm = train_dataset['w_norm']
    batch_size, node_size, emd_size = emd.size()

    # 初始化轨迹存储
    ans_temp = [train_dataset['ans']]                # 解集序列，每个元素形状 (batch, node)
    mask = train_dataset['ans'] & (train_dataset['taboo'] <= 0)  # 初始动作掩码：已选且不在禁忌中的节点
    mask_temp = [mask]                                # 动作掩码序列
    reward = []                                        # 奖励列表，每个元素形状 (batch,)
    p_new = []                                         # 新策略下所选动作的概率（标量），每个元素形状 (batch,) 或更少
    p_old = []                                         # 旧策略下所选动作的概率（标量）
    pi_temp = []                                       # 所选动作的索引（节点编号）
    p_temp = []                                        # 每个时间步的策略分布（完整概率），形状 (batch, node)
    age_temp = [copy.deepcopy(age)]  # 记录每个添加步实际参与计算的样本索引，前两个删除步为全部样本
    index_temp = [torch.arange(batch_size, device=device)] * 2  # 记录每个添加步实际参与计算的样本索引，前两个删除步为全部样本

    # ---------- 第一步删除动作 ----------
    p = models['delete_actor'](emd, train_dataset, ans_temp[-1], mask_temp[-1], age_temp[-1])
    p_temp.append(p)                                   # 保存完整分布
    delete_1 = F.select_node(p, mask_temp[-1], 'sampling')  # 按概率采样选择删除的节点
    pi_temp.append(delete_1)                           # 记录动作索引
    zero_to_batch = torch.arange(batch_size, device=device)
    # 记录所选动作的概率（直接索引）
    p_new.append(torch.zeros(batch_size, device=device) + p[zero_to_batch, delete_1])
    # 更新解集：将删除的节点设为未选中
    ans_temp.append(copy.deepcopy(ans_temp[-1]))
    ans_temp[-1][zero_to_batch, delete_1] = False
    # 更新掩码：删除节点后，该节点从掩码中移除，同时邻居的掩码更新（根据图）
    mask_temp.append(copy.deepcopy(mask_temp[-1]))
    mask_temp[-1][zero_to_batch, delete_1] = False
    #更新配置
    conf = conf | graphs[zero_to_batch, delete_1, :]
    conf[zero_to_batch, delete_1] = False
    # 更新时间
    age_temp.append(copy.deepcopy(age))
    # 奖励 = 删除节点的权重（正奖励，因为减少了总权重）
    reward.append(w_norm[zero_to_batch, delete_1])

    # ---------- 第二步删除动作（类似第一步） ----------
    p = models['delete_actor'](emd, train_dataset, ans_temp[-1], mask_temp[-1], age_temp[-1])
    p_temp.append(p)
    delete_2 = F.select_node(p, mask_temp[-1], 'sampling')
    pi_temp.append(delete_2)
    p_new.append(torch.zeros(batch_size, device=device) + p[zero_to_batch, delete_2])
    ans_temp.append(copy.deepcopy(ans_temp[-1]))
    ans_temp[-1][zero_to_batch, delete_2] = False
    #更新配置
    conf = conf | graphs[zero_to_batch, delete_2, :]
    conf[zero_to_batch, delete_2] = False
    # 更新时间
    age[zero_to_batch, delete_2] = step
    age_temp.append(copy.deepcopy(age))
    
    reward.append(w_norm[zero_to_batch, delete_2])

    # ---------- 添加节点阶段（直到没有未覆盖的边） ----------
    add_node = F.get_no_cov(graphs, ans_temp[-1], conf)      # 获取需要添加的节点（未覆盖且存在未覆盖邻居）
    while add_node.any():
        mask_temp.append(add_node)                      # 将当前可添加节点作为新的掩码
        index_batch = torch.where(add_node.any(dim=-1))[0]  # 哪些样本在该步有可添加节点
        index_temp.append(index_batch)                  # 记录这些样本的索引，供后续使用
        # 调用添加策略网络（仅对需要添加的样本进行计算）
        p = models['add_actor'](emd, train_dataset, ans_temp[-1], mask_temp[-1], age_temp[-1], index_batch)
        
        add_select = F.select_node(p, mask_temp[-1][index_batch], 'sampling')  # 采样选择添加的节点
        pi_temp.append(add_select)
        # 记录所选动作的概率（直接取索引，不填充零）
        p_new.append(p[torch.arange(index_batch.size(0), device=device), add_select])
        # 更新解集：将添加的节点设为选中
        ans_temp.append(copy.deepcopy(ans_temp[-1]))
        ans_temp[-1][index_batch, add_select] = True
        # 更新conf
        conf[index_batch] = conf[index_batch] | graphs[index_batch, add_select, :]
        # 更新时间
        age[index_batch, add_select] = step
        age_temp.append(copy.deepcopy(age))
        # 重新计算未覆盖节点
        add_node = F.get_no_cov(graphs, ans_temp[-1], conf)
        # 奖励 = -添加节点的权重（负奖励，因为增加了总权重）
        r = torch.zeros(batch_size, dtype=torch.float32, device=device)
        r[index_batch] = -w_norm[index_batch, add_select]
        reward.append(r)
        # 保存该步的完整策略分布（仅对有动作的样本计算了分布，但这里保存了p，形状 (len(index_batch), node_size)）
        p_temp.append(p)

    # ---------- 可选删除阶段（启发式，删除冗余节点） ----------
    ok_delete = F.get_ok_delete(graphs, ans_temp[-1])   # 获取可以安全删除的节点（在覆盖中且所有邻居也都在覆盖中）
    while ok_delete.any():
        index_batch = torch.where(ok_delete.any(dim=-1))[0]
        # 使用随机扰动打破平局，选择删除的节点
        ok_delete_int = ok_delete[index_batch].int()
        delete_select = (ok_delete_int.int() + torch.rand(ok_delete_int.size(), device=device) * 0.1).argmax(dim=-1)
        # 更新解集：删除该节点
        ans_temp.append(copy.deepcopy(ans_temp[-1]))
        ans_temp[-1][index_batch, delete_select] = False
        # 重新计算可删除节点
        ok_delete = F.get_ok_delete(graphs, ans_temp[-1])
        # 奖励 = 删除节点的权重（正奖励）
        r = torch.zeros(batch_size, dtype=torch.float32, device=device)
        r[index_batch] = w_norm[index_batch, delete_select]
        reward.append(r)
        # 注意：此阶段没有记录策略分布和动作概率，因为这些删除是启发式的，不参与策略学习

    # ---------- 计算旧策略下所选动作的概率（用于PPO的ratio） ----------
    with torch.no_grad():
        # 前两个删除步使用旧的delete_actor，对所有样本计算概率并索引
        p = models_old['delete_actor'](emd_old, train_dataset, ans_temp[0], mask_temp[0], age_temp[0])
        p_old.append(torch.zeros(batch_size, device=device) + p[zero_to_batch, pi_temp[0]])

        p = models_old['delete_actor'](emd_old, train_dataset, ans_temp[1], mask_temp[1], age_temp[1])
        p_old.append(torch.zeros(batch_size, device=device) + p[zero_to_batch, pi_temp[1]])

        # 后续添加步，根据记录的索引分批计算旧概率
        for i in range(2, len(mask_temp)):
            index_batch = index_temp[i]                # 取出该步实际参与的样本索引
            p = models_old['add_actor'](emd_old, train_dataset, ans_temp[i], mask_temp[i], age_temp[i], index_batch)
            p_old.append(p[torch.arange(index_batch.size(0), device=device), pi_temp[i]])

    # ---------- 验证并更新环境状态（禁忌表、最优解等） ----------
    assert F.check_MVC(graphs, ans_temp[-1]), '该解集不是可行解'
    taboo = ans_temp[-1] & (~train_dataset['ans'])      # 新加入的节点（相对于原始解）
    train_dataset['taboo'] = train_dataset['taboo'] - 1  # 禁忌计数减1
    train_dataset['taboo'][taboo] = opts.taboo_num       # 新加入节点设置禁忌步数
    train_dataset['ans'] = ans_temp[-1]                  # 更新当前解
    w_temp = (ans_temp[-1] * w).sum(dim=-1)              # 计算当前解总权重
    train_dataset['ans_w'] = w_temp
    mask = w_temp < train_dataset['bast_ans_w']          # 若当前解优于历史最优，则更新最优解
    train_dataset['bast_ans'][mask] = ans_temp[-1][mask]
    train_dataset['bast_ans_w'][mask] = w_temp[mask]
    train_dataset['age'] = age_temp[-1]
    
    temp = {'p' :p_temp, 'ans': ans_temp, 'index': index_temp, 'age': age_temp}
    return  reward, temp, p_new,p_old

def compute_ppo_loss(
    emd,                      # 当前状态嵌入，形状 (batch_size, ...)
    critic,                   # 当前critic网络，输入 (emd, ans)
    emd_old,                  # 旧状态嵌入，形状同 emd
    critic_old,               # 旧critic网络
    dataset,                  # 数据集
    reward,                   # 奖励列表，长度 T，每个元素形状 (batch_size)
    p_temp,                   # 策略分布列表，长度 T，每个元素形状 (batch_size, node_size)
    ans_temp,                 # 解集列表，长度 T+1，每个元素形状 (batch_size, ...) 用于critic输入
    p_new,                    # 新策略下所选动作的概率，形状 (T, batch_size)
    p_old,                    # 旧策略下所选动作的概率，形状 (T, batch_size)
    index_temp, 
    age_temp, 
    actor_idx, 
    opts,                     # 可选配置
):
    """
    计算PPO总损失（Actor + Critic + 熵正则）
    返回标量损失
    """
    gamma = opts.gamma  # 折扣因子
    lam = opts.lam  # GAE lambda
    epsilon = opts.epsilon  # PPO裁剪范围
    theta = opts.theta  # 熵系数

    T = len(reward)                 # 轨迹长度
    batch_size = emd.size(0)

    # 1. 计算GAE优势（每个时间步，每个样本）
    # 首先获取所有时间步的旧价值估计
    values_old = []
    for t in range(T+1):            # ans_temp 有 T+1 个（包括初始和最终状态）
        v = critic_old(emd_old, dataset, ans_temp[t], age_temp[min(len(age_temp)-1, t)])   # 形状 (batch_size,)
        values_old.append(v)

    advantages = []
    gae = torch.zeros(batch_size, device=device)
    for t in reversed(range(T)):
        # delta = r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = reward[t] + gamma * values_old[t+1] - values_old[t]
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)               # 按时间顺序存储

    # advantages 现在是列表，每个元素形状 (batch_size,)
    # 计算 returns = advantages + values_old（用于critic目标）
    returns = [adv + values_old[t] for t, adv in enumerate(advantages)]   # 每个形状 (batch_size,)

    # 2. Actor损失（使用裁剪的代理目标）
    advantages_flat=[]
    for i in range(len(actor_idx)):
        advantages_flat.append(advantages[actor_idx[i]][index_temp[i]])
    advantages_flat = torch.cat(advantages_flat, dim=0)   # (T*batch_size,)
    advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std(unbiased=False))
    
    ratio = p_new / (p_old.detach() + 1e-8)    # 避免除零
    surr1 = ratio * advantages_flat.detach()
    surr2 = torch.clamp(ratio, 1-epsilon, 1+epsilon) * advantages_flat.detach()
    actor_loss = -torch.min(surr1, surr2).mean()   # 负号表示梯度上升

    # 3. Critic损失（Huber loss）
    # 使用当前critic估计初始状态的值（或其他时间步？通常所有步都参与，但为简化，只使用初始步）
    # 标准做法是对所有时间步的critic损失求和或平均
    # 这里计算所有时间步的critic损失并平均
    critic_loss = 0.0
    huber_delta = opts.huber_delta
    for t in range(T):
        V_curr = critic(emd, dataset, ans_temp[t], age_temp[min(len(age_temp)-1, t)])        # 当前critic对第t步状态的估计，形状 (batch_size,)
        target = returns[t].detach()                # 目标值，分离梯度
        # critic_loss = (0.5 * (V_curr-target).pow(2)).mean()
        loss = (V_curr-target).abs()
        mask = loss < huber_delta
        critic_loss += (0.5 * mask * loss.pow(2) + ~ mask * (huber_delta * loss - 0.5 * huber_delta ** 2)).mean()
    critic_loss = critic_loss / T                 # 平均每个时间步

    # 4. 熵正则（最大化策略分布的熵）
    entropy = 0.0
    for p_dist in p_temp:                         # p_dist 形状 (batch_size, num_actions)
        # 对分布计算熵，避免 log(0)
        p_dist = p_dist.clamp(min=1e-8)
        entropy_batch = -(p_dist * torch.log(p_dist)).sum(dim=-1)  # 每个样本的熵
        entropy += entropy_batch.mean()            # 平均熵
    entropy = entropy / len(p_temp)                           # 平均每个时间步
    entropy_loss = -theta * entropy                           # 负号：最大化熵
    
    # 总损失
    global flag
    global critic_losses
    global actor_losses
    if flag==1 or flag%50==0 :
        critic_losses.append(critic_loss.item())
        actor_losses.append(actor_loss.item())
        
    total_loss = actor_loss + critic_loss + entropy_loss
    return total_loss
