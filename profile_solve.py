import copy
import random

import numpy as np
import torch

import utils_MWVC.function as F
from options import get_opts
from run import get_models, torch_load_cpu
from utils_MWVC.profiling import CudaSegTimer

CATS = ['encode', 'decode', 'search',
        'move_emd', 'move_emd_assert', 'actor_fwd_delete', 'actor_fwd_add',
        'sample', 'assert_sync']

DECODE_PARTS = ['move_emd', 'move_emd_assert', 'actor_fwd_delete',
                'actor_fwd_add', 'sample', 'assert_sync']


def _collect(timer):
    d = dict(timer.acc_ms)
    d['construct'] = timer.t_construct_ms
    d['total'] = timer.t_total_ms
    return d


def _avg(runs, k):
    return sum(r[k] for r in runs) / len(runs)


def _print_avg(runs):
    n = len(runs)
    tot = _avg(runs, 'total')
    con = _avg(runs, 'construct')
    enc = _avg(runs, 'encode')
    dec = _avg(runs, 'decode')
    sea = _avg(runs, 'search')
    other = max(0.0, tot - con - enc - dec - sea)   # CPU/Python/启动开销 + 已有的逐步同步

    print('\n==== 顶层耗时构成（{} 个实例平均）===='.format(n))
    for name, v in [('construct', con), ('encode', enc), ('decode', dec),
                    ('search', sea), ('other', other)]:
        print('  {:<16s}: {:>9.1f} ms  ({:>5.1f}%)'.format(name, v, 100 * v / tot))
    print('  {:<16s}: {:>9.1f} ms'.format('total', tot))

    # decode 内部拆分（嵌套在 decode 内，故按占 decode 与占 total 两种比例展示）
    parts = [(p, _avg(runs, p)) for p in DECODE_PARTS]
    dec_other = max(0.0, dec - sum(v for _, v in parts))
    parts.append(('decode_other', dec_other))
    print('\n==== decode 内部拆分（嵌套在 decode 内）====')
    for name, v in parts:
        print('  {:<16s}: {:>9.1f} ms  ({:>5.1f}% of decode, {:>5.1f}% of total)'.format(
            name, v, 100 * v / dec, 100 * v / tot))
    print('  {:<16s}: {:>9.1f} ms'.format('decode_total', dec))


def main():
    opts = get_opts()
    opts.use_cuda = torch.cuda.is_available() and not opts.no_cuda
    opts.device = torch.device('cuda' if opts.use_cuda else 'cpu')
    if opts.seed is not None:
        torch.manual_seed(opts.seed)
        np.random.seed(opts.seed)
        random.seed(opts.seed)

    load_data = torch_load_cpu(opts.load_path) if opts.load_path else {}
    if not opts.load_path:
        print('[warn] 未提供 --load_path：随机初始化权重，耗时占比可参考，解质量无意义')
    models = get_models(opts, load_data)
    F.models_eval(models)

    n = opts.n_size

    def make_instance():
        return F.generate_graphs(n, opts.min_p, opts.max_p, 1)   # 单实例 batch=1

    with torch.no_grad():
        # 预热（整体丢弃，触发 cudnn/cublas 初始化）
        warm_opts = copy.copy(opts)
        warm_opts.eval_num = min(5, opts.eval_num)
        F.validate(models, make_instance(), warm_opts, timer=CudaSegTimer(CATS, flush_every=100))
        print('warm-up 完成；开始测量 (n={}, eval_num={}, 3 个实例)'.format(n, opts.eval_num))

        runs = []
        for k in range(3):
            timer = CudaSegTimer(CATS, flush_every=100)
            F.validate(models, make_instance(), opts, timer=timer)
            runs.append(_collect(timer))
            print('  实例 {} 完成: total={:.1f} ms'.format(k + 1, timer.t_total_ms))

    _print_avg(runs)


if __name__ == '__main__':
    main()
