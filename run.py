import json
import os
import pprint as pp
import random
import argparse

# 在导入 torch 之前处理 --no_cuda：清空 CUDA_VISIBLE_DEVICES，
# 使后续所有模块的 torch.cuda.is_available() 一致返回 False。
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument('--no_cuda', action='store_true')
if _pre.parse_known_args()[0].no_cuda:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
import torch
import utils_MWVC.function as F
from DRL_model.Model_WVC import actor_model,critic_model
from DRL_model.encoder import GraphAttentionEncoder
from options import get_opts
from train import train_epoch
from utils_MWVC.function import validate_restart

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def torch_load_cpu(load_path):
    return torch.load(load_path,weights_only=False, map_location=lambda storage, loc: storage)  # Load on CPU

def get_models(opts,load_data):
    encoder=GraphAttentionEncoder(
        node_dim=opts.input_dim,
        embed_dim=opts.embedding_dim, # 嵌入维度
        n_heads=opts.n_heads,
        n_layers=opts.n_encoder_layers,
        normalization=opts.normalization,# 归一化方法
    ).to(opts.device)
    input_dim = opts.embedding_dim + opts.move_dim
    add_actor=actor_model(input_dim = input_dim, 
                          feed_forward_hidden = opts.embedding_dim * 2,
                          hidden_layers = opts.n_decode_layers).to(opts.device)
    
    delete_actor=actor_model(input_dim = input_dim,
                             feed_forward_hidden = opts.embedding_dim * 2,
                             hidden_layers = opts.n_decode_layers).to(opts.device)
    
    critic=critic_model(input_dim = input_dim,
                        feed_forward_hidden = opts.embedding_dim * 2,
                        hidden_layers = opts.n_decode_layers).to(opts.device)
    # 加载状态字典，后面这个是预训练后的参数，load_data中的state与当前的state在key上冲突，则覆盖
    if 'models' in load_data:
        models_data=load_data['models']
        encoder.load_state_dict({**encoder.state_dict(), **models_data.get('encoder', {})})
        add_actor.load_state_dict({**add_actor.state_dict(), **models_data.get('add_actor', {})})
        delete_actor.load_state_dict({**delete_actor.state_dict(), **models_data.get('delete_actor', {})})
        critic.load_state_dict({**critic.state_dict(), **models_data.get('critic', {})})

    return {'encoder':encoder,
            'add_actor':add_actor,
            'delete_actor':delete_actor,
            'critic':critic}

def run(opts):
    # 打印运行参数
    pp.pprint(vars(opts))
    # 设置随机种子
    if opts.seed is not None:
        torch.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)
    os.makedirs(opts.save_dir)
    # 保存参数，这样可以始终找到精确的配置
    with open(os.path.join(opts.save_dir, "args.json"), 'w') as f:
        json.dump(vars(opts), f, indent=True)

    # 设置device
    opts.device = device
    print('使用的设备：{}'.format(opts.device))
    # 根据load_path加载数据
    load_data = {}
    assert opts.load_path is None or opts.resume is None, "Only one of load path and resume can be given"
    load_path = opts.load_path if opts.load_path is not None else opts.resume
    if load_path is not None:
        print('  [*] 加载 {} 中的数据'.format(load_path))
        load_data = torch_load_cpu(load_path)
    # 实例化模型（用于训练）
    models = get_models(opts,load_data)

    # 这里将模型添加到优化器中
    params=[{'params': models['encoder'].parameters(), 'lr': opts.lr_model},
            {'params': models['delete_actor'].parameters(), 'lr': opts.lr_model},
            {'params': models['add_actor'].parameters(), 'lr': opts.lr_model},
            {'params': models['critic'].parameters(), 'lr': opts.lr_model}]
    optim = torch.optim.Adam(params)
    # Load optim state
    if 'optim' in load_data:
        optim.load_state_dict(load_data['optim'])
        # 将优化器的状态数据移动到指定设备
        for state in optim.state.values():
            for k, v in state.items():
                # if isinstance(v, torch.Tensor):
                if torch.is_tensor(v):
                    state[k] = v.to(opts.device)
    # 初始化学习率调度器，根据每个epoch对学习率进行衰减。另外创建验证数据集，数据集大小由图的大小和样本数量决定
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda epoch: opts.lr_decay ** epoch)
    # 开始实际的训练循环
    if opts.dataset is not None:
        with open(opts.dataset, 'r') as f:
            dataset = json.load(f)
        dataset=F.load_dataset_to_tensor(dataset, opts, is_random=False)
    else:
        dataset=F.generate_graphs(opts.n_size,opts.min_p,opts.max_p,opts.graph_size)

    # 继续上次训练没完成的模型
    if opts.resume:
        epoch_resume = int(os.path.splitext(os.path.split(opts.resume)[-1])[0].split("-")[1])

        torch.set_rng_state(load_data['rng_state'])
        if opts.use_cuda:
            torch.cuda.set_rng_state_all(load_data['cuda_rng_state'])
        print("Resuming after {}".format(epoch_resume))
        opts.epoch_start = epoch_resume + 1

    # 初始化和有些参数的保存到此处结束，下面是正式开始
    if opts.eval_only:
        # 只评估
        F.models_eval(models)
        with torch.inference_mode():
            validate_restart(models, dataset, opts, True)
    else:
        torch.autograd.set_detect_anomaly(opts.debug)
        for model in models:
            models[model].train()
        # 以下数据仅用于分析
        dataset['all_ans_w'] = []
        dataset['all_critic_loss'] = []
        dataset['all_actor_loss'] = []
        
        for epoch in range(opts.epoch_start, opts.epoch_start + opts.n_epochs):
            # 训练方法
            train_epoch(
                models,  # 模型
                dataset,    # 测试数据集
                optim,  # 优化器
                opts,
                epoch,
                lr_scheduler,  # 学习率调度器
            )
            # if epoch==10 :
            #     opts.theta=opts.theta_new
        
        with open(os.path.join(opts.save_dir, 'all_ans_w.txt'),'w') as f:
            json.dump(dataset['all_ans_w'],f)
        with open(os.path.join(opts.save_dir, 'all_critic_loss.txt'),'w') as f:
            json.dump(dataset['all_critic_loss'],f)
        with open(os.path.join(opts.save_dir, 'all_actor_loss.txt'),'w') as f:
            json.dump(dataset['all_actor_loss'],f)

if __name__=="__main__":
    run(get_opts())