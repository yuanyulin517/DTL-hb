import numpy as np
import random
import os
import json
import time
from typing import List, Dict, Any

# 假设MVCSolver已经导入
from solver import MWVCSolver  # 裸 sibling 导入：本文件作为脚本从 utils_MWVC/ 内运行
from weighting import sample_weights_np  # 同上，与 solver 一致的 sibling 导入

def generate_graph(n: int, min_edges: int, max_edges: int):
    """
        使用NumPy加速的版本，适合大规模图
    """
    edge_count = random.randint(min_edges, max_edges)
    max_possible = n * (n - 1) // 2

    if edge_count >= max_possible:
        # 生成所有可能的边
        u, v = np.triu_indices(n, k=1)
        return np.column_stack([u, v]).tolist(),max_possible

    # 生成所有可能的边的索引
    all_edge_indices = np.arange(max_possible)

    # 随机选择边索引
    selected_indices = np.random.choice(all_edge_indices, size=edge_count, replace=False)

    # 将索引转换为边
    u, v = np.triu_indices(n, k=1)
    selected_edges = np.column_stack([u[selected_indices], v[selected_indices]])

    return selected_edges.tolist(),edge_count


def generate_graphs(min_n,max_n,min_mult,max_mult,count):
    dataset = []
    for i in range(count):
        # 随机生成顶点数
        n = random.randint(min_n, max_n)
        # 计算边数范围
        min_edges = int(n * min_mult)
        max_edges = int(n * max_mult)
        # 确保边数不超过最大可能边数
        max_possible_edges = n * (n - 1) // 2
        max_edges = min(max_edges, max_possible_edges)
        min_edges = min(min_edges, max_edges)
        # 生成图
        ans_list,e_size = generate_graph(n, min_edges, max_edges)
        #生成顶点权值
        w=sample_weights_np(n).tolist()
        # 添加到数据集
        data_item = {
            "data": ans_list,
            "v_size": n,
            "e_size": e_size,
            "w":w
        }
        dataset.append(data_item)
    return dataset

class MVCDataGenerator:
    def __init__(self):
        self.solver = MWVCSolver()
        self.data_dir = "../data"

        # 创建数据目录
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def save_dataset(self, dataset: List[Dict[str, Any]], filename: str):
        """
        保存数据集到文件
        Args:
            dataset: 数据集
            filename: 文件名
        """
        filepath = os.path.join(self.data_dir, filename)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False)

        print(f"数据集已保存到: {filepath}")

    def generate_complete_dataset(self):
        """
        生成完整的数据集（3000份）
        """
        print("开始生成MWVC数据集...")
        print("=" * 50)

        total_start_time = time.time()

        sol=MWVCSolver()
        # 生成第一部分数据
        print("生成第一部分数据 (300份, 结点数10~40)...")
        part1_start = time.time()
        dataset_part1 = generate_graphs(200,500,1,100,1000)
        dataset_part1 += generate_graphs(500,1000,1,300,1000)
        dataset_part1 += generate_graphs(1000,2000,1,500,1000)
        # i=0
        # for data_item in dataset_part1:
        #     ans,ans_w=sol.solve(data_item['data'],data_item['size'],data_item['w'])
        #     data_item['ans']=ans
        #     data_item['ans_w']=ans_w
        #     data_item['ans_size']=len(ans)
        #     i+=1
        #     if i%100==0:
        #         print("第一部分已完成{}份".format(i))
        # part1_time = time.time() - part1_start
        # print(f"第一部分完成，耗时: {part1_time:.2f}秒")
        # print()
        #
        # # # 生成第二部分数据
        # print("生成第二部分数据 (300份, 结点数41~100)...")
        # part2_start = time.time()
        # dataset_part2 = generate_graphs(41,100,3,20,300)
        # i=0
        # for data_item in dataset_part2:
        #     ans,ans_w=sol.solve(data_item['data'],data_item['size'],data_item['w'])
        #     data_item['ans']=ans
        #     data_item['ans_w']=ans_w
        #     data_item['ans_size']=len(ans)
        #     i+=1
        #     if i%100==0:
        #         print("第二部分已完成{}份".format(i))
        # part2_time = time.time() - part2_start
        # print(f"第二部分完成，耗时: {part2_time:.2f}秒")
        # print()
        #
        # # # 生成第三部分数据
        # print("生成第三部分数据 (750份, 结点数101~500)...")
        # part3_start = time.time()
        # dataset_part3 =generate_graphs(101,500,20,100,300)
        # i=0
        # for data_item in dataset_part3:
        #     ans,ans_w=sol.solve(data_item['data'],data_item['size'],data_item['w'])
        #     data_item['ans']=ans
        #     data_item['ans_w']=ans_w
        #     data_item['ans_size']=len(ans)
        #     i+=1
        #     if i%100==0:
        #         print("第三部分已完成{}份".format(i))
        # part3_time = time.time() - part3_start
        # print(f"第三部分完成，耗时: {part3_time:.2f}秒")
        # print()


        # 合并所有数据
        complete_dataset = dataset_part1 #+ dataset_part2+dataset_part3

        # 保存数据
        print("保存数据集...")
        self.save_dataset(complete_dataset, "text_2000.kpl")

        total_time = time.time() - total_start_time
        print(f"\n数据集生成完成! 总耗时: {total_time:.2f}秒")

        # 统计信息
        # self.print_statistics(complete_dataset)

    def print_statistics(self, dataset: List[Dict[str, Any]]):
        """
        打印数据集统计信息
        """
        sizes = [item["size"] for item in dataset]
        ans_sizes = [item["ans_size"] for item in dataset]

        # 计算边数
        edge_counts = []
        for item in dataset:
            adj_matrix = np.array(item["data"])
            # 无向图，边数是邻接矩阵中1的个数除以2
            edge_count = np.sum(adj_matrix) // 2
            edge_counts.append(edge_count)

        print("\n数据集统计信息:")
        print("=" * 30)
        print(f"总数据量: {len(dataset)}")
        print(f"顶点数范围: {min(sizes)} ~ {max(sizes)}")
        print(f"平均顶点数: {np.mean(sizes):.2f}")
        print(f"边数范围: {min(edge_counts)} ~ {max(edge_counts)}")
        print(f"平均边数: {np.mean(edge_counts):.2f}")
        print(f"解大小范围: {min(ans_sizes)} ~ {max(ans_sizes)}")
        print(f"平均解大小: {np.mean(ans_sizes):.2f}")
        print(f"解大小: {np.sum(ans_sizes)}")

        # 按部分统计
        part1_sizes = [item["size"] for item in dataset[:3200]]
        part2_sizes = [item["size"] for item in dataset[3200:4800]]
        part3_sizes = [item["size"] for item in dataset[4800:]]

        print(f"\n第一部分 (10-30顶点): {len(part1_sizes)}份")
        print(f"第二部分 (31-100顶点): {len(part2_sizes)}份")
        print(f"第三部分 (101-500顶点): {len(part3_sizes)}份")


# 使用示例
if __name__ == "__main__":
    # 创建数据生成器
    generator = MVCDataGenerator()

    # 生成完整数据集
    generator.generate_complete_dataset()

