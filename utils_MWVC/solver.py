from utils_MWVC.fastWVC import FastWVC

class MWVCSolver:
    def __init__(self):
        self.vertices = None
        self.adj_list = None
        self.w=None

    def solve(self, ans_list,n,w):
        """
        根据图规模自动选择算法求解MVC问题
        Args:
            adjacency_matrix: 边列表，二维列表
            n:图规模
        Returns:
            list: 最小顶点覆盖的顶点列表
            :param ans_list:
        """
        # 根据规模选择算法
        if n <= 40:
            self.w=w
            # 将边列表转换为邻接表
            self._build_adj_list(ans_list,n)
            # print("使用精确算法...")
            return self._exact_algorithm()

        else:
            # print("使用fastWVC算法...")
            fw=FastWVC()
            data={'v':w,'e':ans_list,'v_size':len(w),'e_size':len(ans_list)}
            return fw.run_dict(data,1234,10,0)

    def _build_adj_list(self, ans_list,n):
        """将邻接矩阵转换为邻接表"""

        self.vertices = list(range(n))
        self.adj_list = {i:[] for i in range(n)}

        for u,v in ans_list:
            self.adj_list[v].append(u)
            self.adj_list[u].append(v)

    def _exact_algorithm(self):
        """
        修复后的精确算法
        """
        n = len(self.vertices)

        # 预计算所有边
        all_edges = []
        q=0
        for i in range(n):
            if len(self.adj_list[i])==0:
                q|=1<<i
                continue
            for j in self.adj_list[i]:
                if i < j:
                    all_edges.append((i, j))

        # 构建顶点掩码（用于快速邻居检查）
        vertex_masks = [0] * n
        for i in range(n):
            for neighbor in self.adj_list[i]:
                vertex_masks[i] |= (1 << neighbor)

        # 迭代加深搜索
        self.max_w=sum(self.w)
        ans=self._dfs_optimized(0, 0,0.0, set(all_edges), n, vertex_masks,0,q)
        ans_list=[]
        for i in range(n):
            if ans==0 : break
            if (ans&1)==1:
                ans_list.append(i)
            ans>>=1
        return ans_list,self.max_w

    def _dfs_optimized(self, index, current_solution,current_w, uncovered_edges, n, vertex_masks,p,q):
        """
        优化的DFS搜索，使用集合操作
        """
        if (q&(1<<index))!=0:
            return self._dfs_optimized(index+1, current_solution,current_w, uncovered_edges, n, vertex_masks,p,q)

        # 剪枝：当前解已经太大
        if current_w >= self.max_w:
            return None

        # 找到解
        if not uncovered_edges:
            self.max_w=current_w
            return current_solution

        # 处理完所有顶点
        if index >= n:
            return None

        # 分支1：不选择当前顶点
        result=None
        if p&(1<<index)==0:
            new_p=p|vertex_masks[index]
            result = self._dfs_optimized(index + 1, current_solution,current_w, uncovered_edges, n, vertex_masks,new_p,q)

        # 分支2：选择当前顶点
        new_solution = current_solution|(1<<index)
        new_w=current_w+self.w[index]

        # 计算新覆盖的边
        new_covered = set()
        for edge in list(uncovered_edges):
            if index in edge:
                new_covered.add(edge)

        new_uncovered = uncovered_edges - new_covered

        result2 = self._dfs_optimized(index + 1, new_solution,new_w, new_uncovered, n, vertex_masks,p,q)
        return result2 if result2 is not None else result
