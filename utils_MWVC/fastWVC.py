import time
import random
import math
import sys
from typing import List, Tuple


class Edge:
    """边类，表示图中的一条边，连接两个顶点"""
    def __init__(self, v1: int, v2: int):
        self.v1 = v1  # 第一个顶点
        self.v2 = v2  # 第二个顶点


class FastWVC:
    """FastWVC算法实现类，用于求解最小权重顶点覆盖问题"""
    def __init__(self):
        # 算法控制参数
        self.start_time = 0           # 算法开始时间
        self.max_steps = 0            # 最大步数限制
        self.step = 0                 # 当前步数
        self.try_step = 0             # 尝试步数（用于定期检查时间）
        self.seed = 0                 # 随机种子
        self.cutoff_time = 0          # 运行时间限制
        self.mode = 0                 # 算法模式

        # 图结构信息
        self.v_num = 0                # 顶点数量
        self.e_num = 0                # 边数量
        self.edges = []               # 边列表
        self.edge_weight = []         # 边的权重（初始都为1）

        # 顶点相关信息
        self.dscore = []              # dscore值：顶点从覆盖集中移除带来的分数变化
        self.time_stamp = []          # 时间戳：记录顶点最后一次被操作的时间
        self.v_weight = []            # 顶点的权重
        self.v_edges = []             # 与顶点相连的边索引列表
        self.v_adj = []               # 顶点的邻接顶点列表
        self.v_degree = []            # 顶点的度（相邻边数）

        # 当前解的状态
        self.c_size = 0               # 当前覆盖集大小
        self.v_in_c = []              # 顶点是否在覆盖集中，1表示在，0表示不在
        self.remove_cand = []         # 可以移除的顶点候选集
        self.index_in_remove_cand = [] # 顶点在remove_cand中的索引
        self.remove_cand_size = 0     # 可移除候选集的大小
        self.now_weight = 0           # 当前解的权重

        # 最佳解的状态
        self.best_c_size = 0          # 最佳覆盖集大小
        self.best_v_in_c = []         # 最佳覆盖集的顶点状态
        self.best_comp_time = 0       # 找到最佳解的时间
        self.best_step = 0            # 找到最佳解的步数
        self.best_weight = 0          # 最佳解的权重

        # 未覆盖边的管理
        self.uncov_stack = []         # 未覆盖边的栈
        self.uncov_stack_fill_pointer = 0  # 未覆盖边栈的指针
        self.index_in_uncov_stack = []    # 边在未覆盖栈中的索引

        # 算法策略相关
        self.conf_change = []         # 配置变化标记，用于启发式策略
        self.tabu_list = []           # 禁忌表，防止重复操作

        # 权重调整参数
        self.ave_weight = 0           # 边的平均权重
        self.delta_total_weight = 0   # 总权重变化量
        self.threshold = 0            # 权重调整阈值
        self.p_scale = 0.0            # 权重缩放比例

    def build_instance(self, filename: str) -> int:
        """从文件读取图实例，构建数据结构"""
        try:
            with open(filename, 'r') as infile:
                lines = infile.readlines()

            # 解析基本信息（第一行格式：p edge 顶点数 边数）
            parts = lines[0].strip().split()
            self.v_num = int(parts[2])
            self.e_num = int(parts[3])

            # 初始化各种数据结构
            self.edges = [Edge(0, 0) for _ in range(self.e_num)]  # 边列表
            self.edge_weight = [1] * self.e_num  # 边权重初始为1
            self.uncov_stack = [0] * self.e_num  # 未覆盖边栈
            self.index_in_uncov_stack = [0] * self.e_num  # 边在栈中的索引
            self.dscore = [0] * (self.v_num + 1)  # dscore值
            self.time_stamp = [0] * (self.v_num + 1)  # 时间戳
            self.v_edges = [[] for _ in range(self.v_num + 1)]  # 顶点相关的边
            self.v_adj = [[] for _ in range(self.v_num + 1)]  # 邻接顶点
            self.v_degree = [0] * (self.v_num + 1)  # 顶点度
            self.v_weight = [0] * (self.v_num + 1)  # 顶点权重
            self.v_in_c = [0] * (self.v_num + 1)  # 顶点是否在覆盖集中
            self.remove_cand = [0] * (self.v_num + 1)  # 可移除候选
            self.index_in_remove_cand = [0] * (self.v_num + 1)  # 候选索引
            self.best_v_in_c = [0] * (self.v_num + 1)  # 最佳解
            self.conf_change = [1] * (self.v_num + 1)  # 配置变化标记
            self.tabu_list = [0] * (self.v_num + 1)  # 禁忌表

            # 读取顶点权重（格式：v 顶点编号 权重）
            for i in range(1, self.v_num + 1):
                parts = lines[i].strip().split()
                self.v_weight[i] = int(parts[2])

            # 读取边信息（格式：e 顶点1 顶点2）
            for i in range(self.v_num + 1, self.v_num + 1 + self.e_num):
                parts = lines[i].strip().split()
                v1 = int(parts[1])
                v2 = int(parts[2])

                # 更新顶点度
                self.v_degree[v1] += 1
                self.v_degree[v2] += 1

                # 创建边对象
                edge_idx = i - (self.v_num + 1)
                self.edges[edge_idx] = Edge(v1, v2)

            # 构建邻接表
            v_degree_tmp = [0] * (self.v_num + 1)  # 临时计数器

            for e in range(self.e_num):
                v1 = self.edges[e].v1
                v2 = self.edges[e].v2

                # 记录顶点相关的边
                self.v_edges[v1].append(e)
                self.v_edges[v2].append(e)

                # 记录邻接关系
                self.v_adj[v1].append(v2)
                self.v_adj[v2].append(v1)

                # 更新临时计数器
                v_degree_tmp[v1] += 1
                v_degree_tmp[v2] += 1

            return 0  # 成功读取

        except Exception as e:
            print(f"读取文件错误: {e}")
            return 1  # 读取失败

    def reset_remove_cand(self):
        """重置可移除候选集，只包含当前在覆盖集中的顶点"""
        j = 0
        for v in range(1, self.v_num + 1):
            if self.v_in_c[v] == 1:  # 顶点在覆盖集中
                self.remove_cand[j] = v
                self.index_in_remove_cand[v] = j
                j += 1
            else:
                self.index_in_remove_cand[v] = 0
        self.remove_cand_size = j  # 更新候选集大小

    def uncover(self, e: int):
        """将边e标记为未覆盖，加入到未覆盖边栈中"""
        self.uncov_stack[self.uncov_stack_fill_pointer] = e
        self.index_in_uncov_stack[e] = self.uncov_stack_fill_pointer
        self.uncov_stack_fill_pointer += 1

    def cover(self, e: int):
        """将边e标记为已覆盖，从未覆盖边栈中移除"""
        index = self.index_in_uncov_stack[e]
        last_uncov_edge = self.uncov_stack[self.uncov_stack_fill_pointer - 1]
        self.uncov_stack_fill_pointer -= 1
        self.uncov_stack[index] = last_uncov_edge
        self.index_in_uncov_stack[last_uncov_edge] = index

    def add(self, v: int):
        """将顶点v添加到覆盖集中，更新相关状态"""
        self.v_in_c[v] = 1  # 标记为在覆盖集中
        self.c_size += 1  # 覆盖集大小加1
        self.dscore[v] = -self.dscore[v]  # dscore取反
        self.now_weight += self.v_weight[v]  # 更新当前权重

        # 将v添加到可移除候选集中
        self.remove_cand[self.remove_cand_size] = v
        self.index_in_remove_cand[v] = self.remove_cand_size
        self.remove_cand_size += 1

        # 更新与v相邻的顶点和边的状态
        for i in range(len(self.v_edges[v])):
            e = self.v_edges[v][i]  # 边索引
            n = self.v_adj[v][i]  # 相邻顶点

            if self.v_in_c[n] == 0:  # 相邻顶点不在覆盖集中
                self.dscore[n] -= self.edge_weight[e]  # 更新相邻顶点的dscore
                self.conf_change[n] = 1  # 标记配置变化
                self.cover(e)  # 覆盖这条边
            else:  # 相邻顶点在覆盖集中
                self.dscore[n] += self.edge_weight[e]  # 更新相邻顶点的dscore

    def remove(self, v: int):
        """从覆盖集中移除顶点v，更新相关状态"""
        self.v_in_c[v] = 0  # 标记为不在覆盖集中
        self.c_size -= 1  # 覆盖集大小减1
        self.dscore[v] = -self.dscore[v]  # dscore取反
        self.conf_change[v] = 0  # 重置配置变化标记

        # 从可移除候选集中移除v
        last_remove_cand_v = self.remove_cand[self.remove_cand_size - 1]
        index = self.index_in_remove_cand[v]
        self.remove_cand[index] = last_remove_cand_v
        self.index_in_remove_cand[last_remove_cand_v] = index
        self.index_in_remove_cand[v] = 0
        self.remove_cand_size -= 1

        self.now_weight -= self.v_weight[v]  # 更新当前权重

        # 更新与v相邻的顶点和边的状态
        for i in range(len(self.v_edges[v])):
            e = self.v_edges[v][i]  # 边索引
            n = self.v_adj[v][i]  # 相邻顶点

            if self.v_in_c[n] == 0:  # 相邻顶点不在覆盖集中
                self.dscore[n] += self.edge_weight[e]  # 更新相邻顶点的dscore
                self.conf_change[n] = 1  # 标记配置变化
                self.uncover(e)  # 标记边为未覆盖
            else:  # 相邻顶点在覆盖集中
                self.dscore[n] -= self.edge_weight[e]  # 更新相邻顶点的dscore

    def update_target_size(self) -> int:
        """更新目标大小，移除一个顶点并返回移除的顶点"""
        best_remove_v = self.remove_cand[0]
        if self.dscore[best_remove_v] != 0:
            best_dscore = self.v_weight[best_remove_v] / abs(self.dscore[best_remove_v])
        else:
            best_dscore = float('inf')  # 如果dscore为0，设置为无穷大

        # 在可移除候选集中寻找最佳移除顶点
        for i in range(1, self.remove_cand_size):
            v = self.remove_cand[i]
            if self.dscore[v] == 0:
                break
            dscore_v = self.v_weight[v] / abs(self.dscore[v])
            if dscore_v > best_dscore:
                best_dscore = dscore_v
                best_remove_v = v

        self.remove(best_remove_v)  # 移除最佳顶点
        return best_remove_v

    def choose_remove_v(self) -> int:
        """根据启发式规则选择一个顶点进行移除"""
        remove_v = self.remove_cand[random.randint(0, self.remove_cand_size - 1)]
        to_try = min(50, self.remove_cand_size)  # 最多尝试50次

        for _ in range(1, to_try):
            v = self.remove_cand[random.randint(0, self.remove_cand_size - 1)]
            if self.tabu_list[v] == 1:  # 跳过禁忌表中的顶点
                continue

            # 计算顶点的得分
            dscore_v = self.v_weight[v] / abs(self.dscore[v])
            dscore_remove_v = self.v_weight[remove_v] / abs(self.dscore[remove_v])

            if dscore_v < dscore_remove_v:
                continue
            elif dscore_v > dscore_remove_v:
                remove_v = v
            elif self.time_stamp[v] < self.time_stamp[remove_v]:  # 时间戳更早
                remove_v = v

        return remove_v

    def choose_add_from_v(self) -> int:
        """从所有顶点中选择一个顶点添加到覆盖集"""
        add_v = 0
        improvemnt = 0.0

        for v in range(1, self.v_num + 1):
            if self.v_in_c[v] == 1 or self.conf_change[v] == 0:
                continue  # 跳过已在覆盖集中或配置未变化的顶点

            dscore_v = self.dscore[v] / self.v_weight[v]  # 计算收益/成本比
            if dscore_v > improvemnt:
                improvemnt = dscore_v
                add_v = v
            elif dscore_v == improvemnt and add_v != 0 and self.time_stamp[v] < self.time_stamp[add_v]:
                add_v = v  # 收益相同时，选择时间戳更早的顶点

        return add_v

    def choose_add_v(self, remove_v: int, update_v: int = 0) -> int:
        """选择一个顶点添加到覆盖集，优先考虑移除顶点周围"""
        add_v = 0
        improvemnt = 0.0

        # 检查移除顶点的邻居
        for v in self.v_adj[remove_v]:
            if self.v_in_c[v] == 1 or self.conf_change[v] == 0:
                continue

            dscore_v = self.dscore[v] / self.v_weight[v]
            if dscore_v > improvemnt:
                improvemnt = dscore_v
                add_v = v
            elif dscore_v == improvemnt and add_v != 0 and self.time_stamp[v] < self.time_stamp[add_v]:
                add_v = v

        # 检查移除顶点自身
        if self.conf_change[remove_v] == 1 and self.v_in_c[remove_v] == 0:
            dscore_v = self.dscore[remove_v] / self.v_weight[remove_v]
            if dscore_v > improvemnt:
                improvemnt = dscore_v
                add_v = remove_v
            elif dscore_v == improvemnt and add_v != 0 and self.time_stamp[remove_v] < self.time_stamp[add_v]:
                add_v = remove_v

        # 检查更新顶点的邻居（如果有）
        if update_v != 0:
            for v in self.v_adj[update_v]:
                if self.v_in_c[v] == 1 or self.conf_change[v] == 0:
                    continue

                dscore_v = self.dscore[v] / self.v_weight[v]
                if dscore_v > improvemnt:
                    improvemnt = dscore_v
                    add_v = v
                elif dscore_v == improvemnt and add_v != 0 and self.time_stamp[v] < self.time_stamp[add_v]:
                    add_v = v

            # 检查更新顶点自身
            if self.conf_change[update_v] == 1 and self.v_in_c[update_v] == 0:
                dscore_v = self.dscore[update_v] / self.v_weight[update_v]
                if dscore_v > improvemnt:
                    improvemnt = dscore_v
                    add_v = update_v
                elif dscore_v == improvemnt and add_v != 0 and self.time_stamp[update_v] < self.time_stamp[add_v]:
                    add_v = update_v
        if add_v==0:
            print("{}:{}".format(remove_v,update_v))

        return add_v

    def update_best_solution(self):
        """更新最佳解记录"""
        if self.now_weight < self.best_weight:  # 找到更优解
            self.best_v_in_c = self.v_in_c.copy()  # 复制当前解
            self.best_weight = self.now_weight
            self.best_c_size = self.c_size
            self.best_comp_time = self.time_elapsed()
            self.best_step = self.step

    def remove_redundant(self):
        """移除冗余顶点（dscore为0的顶点）"""
        i = 0
        while i < self.remove_cand_size:
            v = self.remove_cand[i]
            if self.v_in_c[v] == 1 and self.dscore[v] == 0:
                self.remove(v)  # 移除冗余顶点
            else:
                i += 1

    def construct_vc(self):
        """构造初始顶点覆盖解"""
        self.uncov_stack_fill_pointer = 0
        self.c_size = 0
        self.best_weight = float('inf')
        self.now_weight = 0

        # 初始化顶点覆盖状态
        self.v_in_c = [0] * (self.v_num + 1)

        # 第一次构造：遍历所有边
        for e in range(self.e_num):
            v1 = self.edges[e].v1
            v2 = self.edges[e].v2

            if self.v_in_c[v1] == 0 and self.v_in_c[v2] == 0:  # 边未覆盖
                # 选择度/权重比更大的顶点加入覆盖集
                v1dd = self.v_degree[v1] / self.v_weight[v1]
                v2dd = self.v_degree[v2] / self.v_weight[v2]
                if v1dd > v2dd:
                    self.v_in_c[v1] = 1
                    self.now_weight += self.v_weight[v1]
                else:
                    self.v_in_c[v2] = 1
                    self.now_weight += self.v_weight[v2]
                self.c_size += 1

        # 保存当前解
        save_v_in_c = self.v_in_c.copy()
        save_c_size = self.c_size
        save_weight = self.now_weight

        # 多次随机构造，寻找更好的初始解
        times = 50
        blocks = list(range((self.e_num - 1) // 1024 + 1))  # 将边分成块

        for _ in range(times):
            self.v_in_c = [0] * (self.v_num + 1)
            self.c_size = 0
            self.now_weight = 0
            random.shuffle(blocks)  # 随机打乱块顺序

            for block in blocks:
                begin = block * 1024
                end = min(begin + 1024, self.e_num)
                idx = list(range(begin, end))  # 当前块的边索引

                while idx:
                    i = random.randint(0, len(idx) - 1)
                    edge_idx = idx[i]
                    v1 = self.edges[edge_idx].v1
                    v2 = self.edges[edge_idx].v2

                    del idx[i]  # 移除已处理的边

                    if self.v_in_c[v1] == 0 and self.v_in_c[v2] == 0:
                        v1dd = self.v_degree[v1] / self.v_weight[v1]
                        v2dd = self.v_degree[v2] / self.v_weight[v2]
                        if v1dd > v2dd:
                            self.v_in_c[v1] = 1
                            self.now_weight += self.v_weight[v1]
                        else:
                            self.v_in_c[v2] = 1
                            self.now_weight += self.v_weight[v2]
                        self.c_size += 1

            # 更新最佳初始解
            if self.now_weight < save_weight:
                save_weight = self.now_weight
                save_c_size = self.c_size
                save_v_in_c = self.v_in_c.copy()

        # 使用最佳初始解
        self.now_weight = save_weight
        self.c_size = save_c_size
        self.v_in_c = save_v_in_c

        # 初始化dscore值
        for e in range(self.e_num):
            v1 = self.edges[e].v1
            v2 = self.edges[e].v2

            if self.v_in_c[v1] == 1 and self.v_in_c[v2] == 0:
                self.dscore[v1] -= self.edge_weight[e]
            elif self.v_in_c[v2] == 1 and self.v_in_c[v1] == 0:
                self.dscore[v2] -= self.edge_weight[e]

        # 重置可移除候选集并移除冗余顶点
        self.reset_remove_cand()
        self.remove_redundant()
        self.update_best_solution()

    def check_solution(self) -> bool:
        """验证当前最佳解是否正确覆盖了所有边"""
        for e in range(self.e_num):
            if self.best_v_in_c[self.edges[e].v1] != 1 and self.best_v_in_c[self.edges[e].v2] != 1:
                print(f", 未覆盖边 {e}")
                return False
        return True

    def forget_edge_weights(self):
        """忘记边权重，重新初始化"""
        new_total_weight = 0

        # 重置dscore
        for v in range(1, self.v_num + 1):
            self.dscore[v] = 0

        # 缩放边权重并重新计算dscore
        for e in range(self.e_num):
            self.edge_weight[e] = int(self.edge_weight[e] * self.p_scale)
            new_total_weight += self.edge_weight[e]

            v1 = self.edges[e].v1
            v2 = self.edges[e].v2

            if self.v_in_c[v1] + self.v_in_c[v2] == 0:  # 边未覆盖
                self.dscore[v1] += self.edge_weight[e]
                self.dscore[v2] += self.edge_weight[e]
            elif self.v_in_c[v1] + self.v_in_c[v2] == 1:  # 边被一个顶点覆盖
                if self.v_in_c[v1] == 1:
                    self.dscore[v1] -= self.edge_weight[e]
                else:
                    self.dscore[v2] -= self.edge_weight[e]

        self.ave_weight = new_total_weight // self.e_num  # 计算平均权重

    def update_edge_weight(self):
        """更新边权重，增加未覆盖边的权重"""
        # 增加所有未覆盖边的权重
        for i in range(self.uncov_stack_fill_pointer):
            e = self.uncov_stack[i]
            self.edge_weight[e] += 1
            self.dscore[self.edges[e].v1] += 1
            self.dscore[self.edges[e].v2] += 1

            # 根据模式更新配置变化标记
            if self.mode % 2 == 1:
                self.conf_change[self.edges[e].v1] = 1
                self.conf_change[self.edges[e].v2] = 1

        # 更新总权重变化
        self.delta_total_weight += self.uncov_stack_fill_pointer

        # 根据模式调整权重
        if self.mode // 2 == 1:
            if self.delta_total_weight >= self.e_num:
                self.ave_weight += 1
                self.delta_total_weight -= self.e_num

            # 如果平均权重达到阈值，重新初始化权重
            if self.ave_weight >= self.threshold:
                self.forget_edge_weights()

    def local_search(self):
        """局部搜索主循环"""
        self.step = 1
        self.try_step = 100  # 每100步检查一次时间

        # 初始化权重调整参数
        self.ave_weight = 1
        self.delta_total_weight = 0
        self.p_scale = 0.3
        self.threshold = int(0.5 * self.v_num)

        while True:
            self.update_best_solution()  # 更新最佳解
            update_v = self.update_target_size()  # 移除一个顶点

            # 定期检查时间限制
            if self.step % self.try_step == 0:
                if self.time_elapsed() >= self.cutoff_time:
                    return  # 超时退出

            # 选择并移除一个顶点
            remove_v = self.choose_remove_v()
            self.remove(remove_v)
            self.time_stamp[remove_v] = self.step  # 更新时间戳

            self.tabu_list = [0] * (self.v_num + 1)  # 重置禁忌表

            # 处理所有未覆盖边
            while self.uncov_stack_fill_pointer > 0:
                add_v = self.choose_add_v(remove_v, update_v)  # 选择添加顶点
                self.add(add_v)  # 添加顶点
                self.update_edge_weight()  # 更新边权重
                self.tabu_list[add_v] = 1  # 标记为禁忌
                self.time_stamp[add_v] = self.step  # 更新时间戳

            self.remove_redundant()  # 移除冗余顶点
            self.step += 1
            update_v = 0  # 重置更新顶点

    def time_elapsed(self) -> float:
        """计算从开始到现在的运行时间"""
        return time.time() - self.start_time

    def run(self, filename: str, seed: int, cutoff_time: int, mode: int):
        """运行FastWVC算法的主函数"""
        self.seed = seed
        self.cutoff_time = cutoff_time
        self.mode = mode

        random.seed(seed)  # 设置随机种子

        # 读取图实例
        if self.build_instance(filename) != 0:
            print("打开实例文件失败。")
            return

        print(filename, end="")  # 输出文件名

        self.start_time = time.time()  # 记录开始时间

        # 构造初始解并进行局部搜索
        self.construct_vc()
        self.local_search()

        # 验证并输出结果
        if self.check_solution():
            print(f", {self.best_weight}, {self.best_comp_time}")
        else:
            print(", 解不正确。")

    def build_from_dict(self, data: dict) -> int:
        """从字典数据构建图实例"""
        try:
            # 从字典获取图信息
            self.v_num = data.get('v_size', 0)  # 顶点数
            self.e_num = data.get('e_size', 0)  # 边数
            v_weight_list = data.get('v', [])  # 顶点权重列表，从0开始
            edges_list = data.get('e', [])  # 边列表，从0开始

            # 验证数据
            if self.v_num <= 0 or self.e_num <= 0:
                return 1
            if len(v_weight_list) != self.v_num or len(edges_list) != self.e_num:
                return 1

            # 初始化各种数据结构（与build_instance相同）
            self.edges = [Edge(0, 0) for _ in range(self.e_num)]
            self.edge_weight = [1] * self.e_num
            self.uncov_stack = [0] * self.e_num
            self.index_in_uncov_stack = [0] * self.e_num
            self.dscore = [0] * (self.v_num + 1)
            self.time_stamp = [0] * (self.v_num + 1)
            self.v_edges = [[] for _ in range(self.v_num + 1)]
            self.v_adj = [[] for _ in range(self.v_num + 1)]
            self.v_degree = [0] * (self.v_num + 1)
            self.v_weight = [0] * (self.v_num + 1)
            self.v_in_c = [0] * (self.v_num + 1)
            self.remove_cand = [0] * (self.v_num + 1)
            self.index_in_remove_cand = [0] * (self.v_num + 1)
            self.best_v_in_c = [0] * (self.v_num + 1)
            self.conf_change = [1] * (self.v_num + 1)
            self.tabu_list = [0] * (self.v_num + 1)

            # 设置顶点权重（注意：内部存储从索引1开始）
            for i in range(self.v_num):
                self.v_weight[i + 1] = v_weight_list[i]

            # 读取边信息（注意：内部存储从索引1开始）
            for i in range(self.e_num):
                v1, v2 = edges_list[i]
                # 验证顶点编号在有效范围内
                if v1 < 0 or v1 >= self.v_num or v2 < 0 or v2 >= self.v_num:
                    print(f"错误: 边 {i} 的顶点编号超出范围: ({v1}, {v2})")
                    return 1

                # 转换为内部表示（从1开始）
                v1_internal = v1 + 1
                v2_internal = v2 + 1

                # 更新顶点度
                self.v_degree[v1_internal] += 1
                self.v_degree[v2_internal] += 1

                # 存储边
                self.edges[i] = Edge(v1_internal, v2_internal)

            # 构建邻接表

            for e in range(self.e_num):
                v1 = self.edges[e].v1
                v2 = self.edges[e].v2

                self.v_edges[v1].append(e)
                self.v_edges[v2].append(e)

                self.v_adj[v1].append(v2)
                self.v_adj[v2].append(v1)

            # print(f"从字典构建图成功: {self.v_num} 个顶点, {self.e_num} 条边")
            return 0

        except Exception as e:
            print(f"从字典构建图实例时发生错误: {e}")
            return 1

    def run_dict(self, data: dict, seed: int, cutoff_time: int, mode: int):
        """直接从字典数据运行FastWVC算法"""
        self.seed = seed
        self.cutoff_time = cutoff_time
        self.mode = mode

        random.seed(seed)  # 设置随机种子

        # 从字典构建图实例
        if self.build_from_dict(data) != 0:
            print("从字典构建图实例失败。")
            return None

        # print(f"正在运行FastWVC算法: seed={seed}, cutoff_time={cutoff_time}, mode={mode}")

        self.start_time = time.time()  # 记录开始时间

        # 构造初始解并进行局部搜索
        self.construct_vc()
        self.local_search()

        # 验证解的正确性
        if not self.check_solution():
            print("解不正确。")
            return None

        # 将结果转换为从0开始的顶点编号列表
        ans = []
        for v in range(1, self.v_num + 1):
            if self.best_v_in_c[v] == 1:  # 顶点在最佳覆盖集中
                ans.append(v - 1)  # 转换回从0开始的编号

        # print(f"找到覆盖集: {len(ans)} 个顶点, 总权重: {self.best_weight}, 耗时: {self.best_comp_time:.6f}秒")
        return ans,self.best_weight

def main():
    """主函数，处理命令行参数并运行算法"""
    if len(sys.argv) == 1:  # 没有参数
        print("FastWVC - 最小权重顶点覆盖问题求解器。")
        print("用法: python mwvc.py [图文件] [随机种子] [截止时间] [CC模式]")
        return 1

    if len(sys.argv) < 5:  # 参数不足
        print("缺少参数。")
        print("用法: python mwvc.py [图文件] [随机种子] [截止时间] [CC模式]")
        return 1

    filename = sys.argv[1]  # 图文件路径

    try:
        seed = int(sys.argv[2])  # 随机种子
        cutoff_time = int(sys.argv[3])  # 时间限制
        mode = int(sys.argv[4])  # 算法模式
    except ValueError:
        print("无效参数。")
        return 1

    # 参数合法性检查并设置默认值
    if seed < 0 or seed > 0xFFFFFFFF:
        seed = 10
    if cutoff_time < 0 or cutoff_time > 0x7FFFFFFF:
        cutoff_time = 1000
    if mode < 0 or mode > 3:
        mode = 0
    with open(filename, 'r') as infile:
        lines = infile.readlines()
    data={}
    parts = lines[0].strip().split()
    data['v_size'] = int(parts[2])
    data['e_size'] = int(parts[3])
    # 读取顶点权重（格式：v 顶点编号 权重）
    data['v']=[0 for i in range(data['v_size'])]
    for i in range(1, data['v_size'] + 1):
        parts = lines[i].strip().split()
        data['v'][i-1] = int(parts[2])

    # 读取边信息（格式：e 顶点1 顶点2）
    data['e']=[(0,0) for i in range(data['e_size'])]
    for i in range(data['v_size'] + 1, data['v_size'] + 1 + data['e_size']):
        parts = lines[i].strip().split()
        v1 = int(parts[1])-1
        v2 = int(parts[2])-1
        data['e'][i-data['v_size']-1]=(v1, v2)
    # 创建求解器并运行
    solver = FastWVC()
    # solver.run(filename, seed, cutoff_time, mode)
    solver.run_dict(data, seed, cutoff_time, mode)

    return 0


if __name__ == "__main__":
    main()