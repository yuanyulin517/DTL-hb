"""
reevaluate_checkpoints.py —— 用修正后的 validate 重评估单个 seed 的全部 checkpoint。

背景：旧的训练期验证分是用带 bug 的 validate 打的，已作废（bug 修复见 commit 48cbffb）。
本脚本遍历某个 seed 的 checkpoint 目录下所有 epoch-N.pt，逐个加载权重，用修正后的
validate（deterministic=True、固定验证集 is_random=False、指定 eval_num）重新评估，
得到每个 epoch 在验证集上的指标（50 张图的平均最优覆盖权重，越低越优），写入
reeval_s{seed}.txt，最后 argmin 选出新的最优 epoch。

这是【评估】脚本，纯推理（torch.inference_mode），不更新任何权重、不改训练代码、不重新训练。
评估路径完全复用项目现有函数（get_models / load_dataset_to_tensor / models_eval / validate），
与修正后训练期验证调用（train.py 中 F.models_eval(models) + validate(..., deterministic=True)）一致。
"""

import os
import re
import csv
import glob
import json
import argparse

import torch

import utils_MWVC.function as F
from options import get_opts
# 复用 run.py 的模型构建/加载逻辑，确保与训练/评估一致
from run import get_models, torch_load_cpu, device


def parse_args():
    p = argparse.ArgumentParser(description="用修正后的 validate 重评估单个 seed 的全部 checkpoint")
    p.add_argument('--seed', type=int, required=True, help='随机种子，如 1234')
    p.add_argument('--ckpt_dir', type=str, default=None,
                   help='checkpoint 目录，如 outputs/MWVC/baseline_s1234_xxx/；'
                        '不传则按 outputs/MWVC/baseline_s{seed}_* 自动匹配')
    p.add_argument('--dataset', type=str, default='data/val_er_n100_d10_seed9100.kpl',
                   help='验证数据集文件（固定验证集）')
    p.add_argument('--eval_num', type=int, default=128, help='评估步数')
    p.add_argument('--output_root', type=str, default='outputs/MWVC',
                   help='自动匹配 ckpt_dir 时的搜索根目录')
    return p.parse_args()


def resolve_ckpt_dir(args):
    if args.ckpt_dir is not None:
        if not os.path.isdir(args.ckpt_dir):
            raise FileNotFoundError('指定的 --ckpt_dir 不存在: {}'.format(args.ckpt_dir))
        return args.ckpt_dir
    pattern = os.path.join(args.output_root, 'baseline_s{}_*'.format(args.seed))
    matches = [d for d in glob.glob(pattern) if os.path.isdir(d)]
    if len(matches) == 0:
        raise FileNotFoundError('未找到匹配的 checkpoint 目录: {}'.format(pattern))
    if len(matches) > 1:
        raise RuntimeError('匹配到多个 checkpoint 目录，请用 --ckpt_dir 明确指定:\n  ' +
                           '\n  '.join(sorted(matches)))
    return matches[0]


def list_checkpoints(ckpt_dir):
    """返回按 epoch 数值升序排列的 [(epoch, path), ...]。"""
    items = []
    for path in glob.glob(os.path.join(ckpt_dir, 'epoch-*.pt')):
        m = re.search(r'epoch-(\d+)\.pt$', os.path.basename(path))
        if m:
            items.append((int(m.group(1)), path))
    items.sort(key=lambda x: x[0])
    return items


def build_opts(args, ckpt_dir):
    """构建 opts：默认值 + ckpt 目录里的 args.json（若有）覆盖，再钉死本次评估相关字段。"""
    opts = get_opts([])  # 解析默认值，不读 sys.argv，也不创建任何目录
    args_json = os.path.join(ckpt_dir, 'args.json')
    if os.path.isfile(args_json):
        # 用训练时保存的配置覆盖（架构相关参数必须与 checkpoint 匹配）
        with open(args_json, 'r') as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            setattr(opts, k, v)
        print('  [*] 已加载训练配置: {}'.format(args_json))
    else:
        print('  [!] 未找到 args.json，使用 options.py 默认架构参数（请确认与训练一致）')
    # 钉死本次评估相关字段
    opts.dataset = args.dataset
    opts.eval_num = args.eval_num
    opts.seed = args.seed
    opts.device = device
    opts.use_cuda = torch.cuda.is_available()
    return opts


def load_val_dataset(opts):
    with open(opts.dataset, 'r') as f:
        raw = json.load(f)
    # is_random=False：与训练里固定验证集的加载方式一致（不打乱节点映射）
    dataset = F.load_dataset_to_tensor(raw, opts, is_random=False)
    return dataset, len(raw)


def evaluate_checkpoint(ckpt_path, dataset, opts):
    """加载一个 checkpoint 的权重，跑修正后的 validate，返回验证指标（越低越优）。"""
    load_data = torch_load_cpu(ckpt_path)
    models = get_models(opts, load_data)   # 复用 run.py 的加载逻辑
    F.models_eval(models)
    with torch.inference_mode():
        # deterministic=True：贪心解码 + 固定构造随机性，与训练期 checkpoint 选择口径一致
        best_w = F.validate(models, dataset, opts, is_print=False, deterministic=True)
        metric = best_w.mean().item()
    return metric


def main():
    args = parse_args()
    ckpt_dir = resolve_ckpt_dir(args)
    opts = build_opts(args, ckpt_dir)

    print('seed={}  ckpt_dir={}'.format(args.seed, ckpt_dir))
    print('dataset={}  eval_num={}  device={}'.format(opts.dataset, opts.eval_num, opts.device))

    checkpoints = list_checkpoints(ckpt_dir)
    if not checkpoints:
        raise FileNotFoundError('在 {} 下未找到 epoch-*.pt'.format(ckpt_dir))

    dataset, n_graphs = load_val_dataset(opts)
    print('验证集图数量：{}'.format(n_graphs))
    print('待评估 checkpoint 数量：{}（epoch {}..{}）\n'.format(
        len(checkpoints), checkpoints[0][0], checkpoints[-1][0]))

    out_path = 'reeval_s{}.txt'.format(args.seed)
    results = []  # [(epoch, metric)]
    with open(out_path, 'w', newline='') as fout:
        writer = csv.writer(fout)
        writer.writerow(['epoch', 'metric'])
        fout.flush()
        for i, (epoch, path) in enumerate(checkpoints):
            metric = evaluate_checkpoint(path, dataset, opts)
            results.append((epoch, metric))
            writer.writerow([epoch, '{:.6f}'.format(metric)])
            fout.flush()  # 增量落盘，部分进度可保留
            print('[{:>4}/{}] epoch={:<5} 验证指标={:.6f}'.format(
                i + 1, len(checkpoints), epoch, metric))

    best_epoch, best_metric = min(results, key=lambda x: x[1])
    print('\n结果已写入 {}'.format(out_path))
    print('seed {}: 新最优 epoch={}, 验证指标={:.6f}'.format(args.seed, best_epoch, best_metric))


if __name__ == '__main__':
    main()
