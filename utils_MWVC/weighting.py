# ER/BA 图节点赋权的唯一来源，消除各生成处 randint 上界漂移
# （原 generate_graphs 用 (20,120)，data_create 用 (20,100)，此处统一到 (20,120)）。
# randint 约定：区间 [W_LOW, W_HIGH) 左闭右开，与原 torch.randint(20, 120) 一致。
# torch/numpy 采用惰性导入：data_create 只需 numpy，不会因此被迫加载 torch。

W_LOW = 20
W_HIGH = 120


def sample_weights_torch(shape, device):
    """生成 torch float32 权重张量，形状 shape，落在 [W_LOW, W_HIGH)。"""
    import torch
    return torch.randint(W_LOW, W_HIGH, shape, dtype=torch.float32, device=device)


def sample_weights_np(n):
    """生成长度为 n 的 numpy 整数权重数组，落在 [W_LOW, W_HIGH)。"""
    import numpy as np
    return np.random.randint(W_LOW, W_HIGH, n)
