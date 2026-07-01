import random
import time
import sys
from typing import List, Tuple

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ConstructWVC:
    def __init__(self, w, v1_idx, v2_idx):
        # 图的基本信息
        self.v_num: int = 0  # 顶点数（顶点编号从1开始）
        self.e_num: int = 0  # 边数

        # 边相关数据
        self.edge: List[Tuple[int, int]] = []  # 边列表：(v1, v2)

        # 顶点相关数据
        self.v_weight: List[int] = []  # 顶点权重
        self.v_adj: List[List[int]] = []  # 邻接表：v_adj[v]是v的邻接顶点列表
        self.v_edges: List[List[int]] = []  # 顶点关联边索引：v_edges[v]是v的边索引列表
        self.v_degree: List[int] = []  # 顶点度数

        # 当前解状态
        self.c_size: int = 0  # 当前覆盖集大小
        self.v_in_c: List[bool] = []  # 顶点是否在覆盖集：1=在，0=不在
        self.remove_cand: List[int] = []  # 覆盖集候选（待移除顶点）
        self.index_in_remove_cand: List[int] = []  # 顶点在remove_cand中的索引
        self.remove_cand_size: int = 0  # remove_cand的大小
        self.now_weight: int = 0  # 当前覆盖集总权重

        # 最优解状态
        self.best_c_size: int = 0
        self.best_v_in_c: List[bool] = []
        self.best_comp_time: float = 0.0
        self.best_step: int = 0
        self.best_weight: int = 0

        # 未覆盖边的栈（数组模拟）
        self.uncov_stack: List[int] = []
        self.uncov_stack_fill_pointer: int = 0
        self.index_in_uncov_stack: List[int] = []

        # 配置检查（CC）与禁忌表
        self.dscore: List[int] = []  # 顶点分数（核心决策依据）
        self.conf_change: List[int] = []  # 配置变化标记（CC策略）
        self.tabu_list: List[int] = []  # 禁忌表

        # 算法控制参数
        self.step: int = 0
        self.try_step: int = 0
        self.seed: int = 0
        self.cutoff_time: int = 0
        self.mode: int = 0
        self.start_time: float = 0.0

        self.BuildInstance(w, v1_idx, v2_idx)

    def TimeElapsed(self) -> float:
        """计算从开始到当前的耗时（秒）"""
        return time.perf_counter() - self.start_time

    def BuildInstance(self, w, v1_idx, v2_idx):
        self.v_num = int(w.shape[0])

        # 步骤1：先收集所有v1<v2的有效边，再赋值e_num
        edge_list = []
        for i in range(v1_idx.shape[0]):
            v1 = int(v1_idx[i])
            v2 = int(v2_idx[i])
            if v1 < v2:
                edge_list.append((v1, v2))
        self.e_num = len(edge_list)  # 实际有效边数
        self.edge = edge_list  # 直接赋值有效边，无需初始化为(0,0)

        # 后续初始化调整：去掉原有的edge_ptr遍历，直接用edge_list
        self.uncov_stack = [0] * self.e_num
        self.index_in_uncov_stack = [-1] * self.e_num

        self.dscore = [0] * self.v_num
        self.v_adj = [[] for _ in range(self.v_num)]
        self.v_edges = [[] for _ in range(self.v_num)]
        self.v_degree = [0] * self.v_num
        self.v_weight = [0] * self.v_num
        self.v_in_c = [False] * self.v_num
        self.remove_cand = [-1] * self.v_num
        self.index_in_remove_cand = [-1] * self.v_num
        self.best_v_in_c = [False] * self.v_num
        self.conf_change = [1] * self.v_num
        self.tabu_list = [0] * self.v_num

        # 读取顶点权重
        for v in range(self.v_num):
            self.v_weight[v] = int(w[v])

        # 统计度数、填充邻接表和边索引表（直接遍历有效边）
        for e in range(self.e_num):
            v1, v2 = self.edge[e]
            self.v_degree[v1] += 1
            self.v_degree[v2] += 1
            self.v_edges[v1].append(e)
            self.v_edges[v2].append(e)
            self.v_adj[v1].append(v2)
            self.v_adj[v2].append(v1)


    def ResetRemoveCand(self):
        """重置覆盖集候选集"""
        j = 0
        for v in range(0, self.v_num):
            if self.v_in_c[v]:
                self.remove_cand[j] = v
                self.index_in_remove_cand[v] = j
                j += 1
            else:
                self.index_in_remove_cand[v] = -1
        self.remove_cand_size = j


    def Uncover(self, e: int):
        """将边标记为未覆盖（压入栈）"""
        self.index_in_uncov_stack[e] = self.uncov_stack_fill_pointer
        self.uncov_stack[self.uncov_stack_fill_pointer] = e
        self.uncov_stack_fill_pointer += 1


    def Cover(self, e: int):
        """将边标记为覆盖（交换弹出）"""
        # 弹出栈顶
        self.uncov_stack_fill_pointer -= 1
        last_uncov_edge = self.uncov_stack[self.uncov_stack_fill_pointer]
        # 交换当前边与栈顶边的位置
        idx = self.index_in_uncov_stack[e]
        self.uncov_stack[idx] = last_uncov_edge
        self.index_in_uncov_stack[last_uncov_edge] = idx


    def Add(self, v: int):
        if v == -1: return
        """将顶点v添加到覆盖集"""
        self.v_in_c[v] = True
        self.c_size += 1
        self.now_weight += self.v_weight[v]

        # 加入移除候选集
        self.remove_cand[self.remove_cand_size] = v
        self.index_in_remove_cand[v] = self.remove_cand_size
        self.remove_cand_size += 1

        # 遍历关联边
        edge_count = len(self.v_edges[v])
        for i in range(edge_count):
            e = self.v_edges[v][i]
            n = self.v_adj[v][i]
            self.dscore[n] -= 1
            if not self.v_in_c[n]:
                self.conf_change[n] = 1
                self.Cover(e)


    def Remove(self, v: int):
        if v == -1: return
        """将顶点v从覆盖集移除"""
        self.v_in_c[v] = False
        self.c_size -= 1
        self.conf_change[v] = 0

        # 从候选集中删除（交换最后一个元素）
        self.remove_cand_size -= 1
        last_v = self.remove_cand[self.remove_cand_size]
        idx = self.index_in_remove_cand[v]
        self.remove_cand[idx] = last_v
        self.index_in_remove_cand[last_v] = idx
        self.index_in_remove_cand[v] = -1

        self.now_weight -= self.v_weight[v]

        # 遍历关联边
        edge_count = len(self.v_edges[v])
        for i in range(edge_count):
            e = self.v_edges[v][i]
            n = self.v_adj[v][i]
            self.dscore[n] += 1
            if not self.v_in_c[n]:
                self.conf_change[n] = 1
                self.Uncover(e)


    def UpdateBestSolution(self):
        """更新最优解"""
        if self.now_weight < self.best_weight:
            self.best_v_in_c = self.v_in_c.copy()
            self.best_weight = self.now_weight
            self.best_c_size = self.c_size
            self.best_comp_time = self.TimeElapsed()
            self.best_step = self.step


    def RemoveRedundant(self):
        """移除冗余顶点（dscore=0）"""
        i = 0
        while i < self.remove_cand_size:
            v = self.remove_cand[i]
            if self.v_in_c[v] and self.dscore[v] == 0:
                self.Remove(v)
                i -= 1  # 索引回退
            i += 1

    def ConstructVC(self):
        """构造初始顶点覆盖（分块随机贪心）"""
        self.uncov_stack_fill_pointer = 0
        self.c_size = 0
        self.best_weight = sys.maxsize
        self.now_weight = 0

        # 初始贪心构造解
        for e in range(self.e_num):
            v1, v2 = self.edge[e]
            if not self.v_in_c[v1] and not self.v_in_c[v2]:
                # 度数/权重比
                v1dd = self.v_degree[v1] / self.v_weight[v1]
                v2dd = self.v_degree[v2] / self.v_weight[v2]
                if v1dd > v2dd:
                    self.v_in_c[v1] = True
                    self.now_weight += self.v_weight[v1]
                else:
                    self.v_in_c[v2] = True
                    self.now_weight += self.v_weight[v2]
                self.c_size += 1

        # 保存初始解
        save_v_in_c = self.v_in_c.copy()
        save_c_size = self.c_size
        save_weight = self.now_weight

        # 分块随机多次尝试
        times = 50
        block_size = 1024
        blocks = list(range((self.e_num + block_size - 1) // block_size))

        for _ in range(times):
            # 重置状态
            self.v_in_c = [False] * self.v_num
            self.c_size = 0
            self.now_weight = 0
            # 打乱块顺序
            random.shuffle(blocks)

            for block in blocks:
                begin = block * block_size
                end = min(begin + block_size, self.e_num)
                idx = list(range(begin, end))
                tmpsize = len(idx)
                while tmpsize > 0:
                    # 随机选择边
                    i = random.randint(0, tmpsize - 1)
                    e_idx = idx[i]
                    # 交换并缩小范围
                    idx[i], idx[tmpsize - 1] = idx[tmpsize - 1], idx[i]
                    tmpsize -= 1
                    # 处理边
                    v1, v2 = self.edge[e_idx]
                    if not self.v_in_c[v1] and not self.v_in_c[v2]:
                        v1dd = self.v_degree[v1] / self.v_weight[v1]
                        v2dd = self.v_degree[v2] / self.v_weight[v2]
                        if v1dd > v2dd:
                            self.v_in_c[v1] = True
                            self.now_weight += self.v_weight[v1]
                        else:
                            self.v_in_c[v2] = True
                            self.now_weight += self.v_weight[v2]
                        self.c_size += 1

            # 更新最优初始解
            if self.now_weight < save_weight:
                save_weight = self.now_weight
                save_c_size = self.c_size
                save_v_in_c = self.v_in_c.copy()

        # 恢复最优初始解
        self.v_in_c = save_v_in_c
        self.c_size = save_c_size
        self.now_weight = save_weight

        # 初始化dscore
        for e in range(self.e_num):
            v1, v2 = self.edge[e]
            if self.v_in_c[v1] and not self.v_in_c[v2]:
                self.dscore[v1] += 1
            elif self.v_in_c[v2] and not self.v_in_c[v1]:
                self.dscore[v2] += 1

        # 重置候选集并移除冗余顶点
        self.ResetRemoveCand()
        for v in range(0, self.v_num):
            if self.v_in_c[v] and self.dscore[v] == 0:
                self.Remove(v)

        self.UpdateBestSolution()

    def CheckSolution(self) -> int:
        """验证最优解是否覆盖所有边"""
        for e in range(self.e_num):
            v1, v2 = self.edge[e]
            if not self.best_v_in_c[v1] and not self.best_v_in_c[v2]:
                print(f"未覆盖边：{e} ({v1}, {v2})")
                return 0
        return 1

    def Reset_tabu_list(self):
        self.tabu_list = [0] * self.v_num  # 重置为0（非禁忌）

    def get_ans(self):
        return self.best_v_in_c

    def get_ans_w(self):
        return self.best_weight


