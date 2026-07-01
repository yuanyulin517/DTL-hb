# 生成固定验证集：50 张 ER 图，n=100，固定期望平均度 d̄≈10（p=10/(n-1)≈0.101）。
# 边模型 = G(n,p) 伯努利，与训练生成器 function.generate_graphs 同分布族：
#   每个无序点对 (i<j) 以概率 p 独立成边（与 train 侧“严格下三角 rand<p 后对称化”等价）。
# 赋权沿用 weighting.sample_weights_np（W_LOW=20, W_HIGH=120），与训练口径一致。
# 输出为 loader 可读的 JSON 数组（每项 {"data","v_size","e_size","w"}），存到 data/。
#
# 从 utils_MWVC/ 内运行：  python make_val_set.py
# 只 import weighting，避免拉起 solver/MWVCSolver 重依赖。

import os
import json
import numpy as np

from weighting import sample_weights_np  # 与 data_create 一致的 sibling 导入

SEED = 9100          # 与训练 seed 1234/2345/3456、将来测试集 seed 均不同
COUNT = 50           # 验证图张数
N = 100              # 每张图节点数
D_BAR = 10.0         # 目标期望平均度
P = D_BAR / (N - 1)  # 反解边概率 ≈ 0.10101
OUT_NAME = "val_er_n100_d10_seed9100.kpl"


def gen_one_graph(n, p):
    """G(n,p) 伯努利：返回 (边表 list[[u,v]] u<v 0索引, 边数)。"""
    u, v = np.triu_indices(n, k=1)          # 所有无序点对，u<v
    keep = np.random.rand(u.shape[0]) < p   # 每对一次伯努利抽样
    edges = np.column_stack([u[keep], v[keep]])
    return edges.tolist(), int(keep.sum())


def main():
    np.random.seed(SEED)

    dataset = []
    for _ in range(COUNT):
        edges, e_size = gen_one_graph(N, P)
        w = sample_weights_np(N).tolist()
        dataset.append({
            "data": edges,
            "v_size": N,
            "e_size": e_size,
            "w": w,
        })

    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, OUT_NAME)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False)

    # 统计核对
    e_list = np.array([d["e_size"] for d in dataset])
    w_all = np.array([wi for d in dataset for wi in d["w"]])
    print(f"已保存: {os.path.normpath(out_path)}")
    print(f"seed={SEED}  count={COUNT}  n={N}  p={P:.6f}  (target avg_deg={D_BAR})")
    print(f"边数: mean={e_list.mean():.1f}  min={e_list.min()}  max={e_list.max()}  (期望 {P * N * (N - 1) / 2:.1f})")
    print(f"实际平均度 avg_deg: mean={2 * e_list.mean() / N:.3f}  min={2 * e_list.min() / N:.3f}  max={2 * e_list.max() / N:.3f}")
    print(f"权重: range=[{w_all.min()}, {w_all.max()}]  mean={w_all.mean():.2f}  (区间 [20,120))")


if __name__ == "__main__":
    main()
