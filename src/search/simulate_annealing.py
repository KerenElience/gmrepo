import random
import math
from copy import deepcopy
from gmrepo.src.evaluator import DIestimator
from gmrepo.src.utils import random_partition

class SimulatedAnnealing():
    def __init__(self, disease: list, estimator: DIestimator, min_size: int = 2, 
                 max_size: int = 5,
                 initial_temp: float = 5.0,
                 cooling_rate: float = 0.95,
                 iteration: int = 100):
        self.insolution = random_partition(disease, min_size, max_size)
        self.T = initial_temp
        self.cooling_rate = cooling_rate
        self.min_size = min_size
        self.max_size = max_size
        self.iteration = iteration
        self.energy_log = []

        self.estimator = estimator

    def _get_energy(self, group: list):
        """快速计算一组的 Recall"""
        score = self.estimator.get_metrics(group)
        self.energy_log.append(score)
        return score

    def _calculate_energy(self, groups: list):
        """计算总能量 (即总 Recall)"""
        total_recall = 0
        
        for group in groups:
            r = self._get_energy(group)
            total_recall += r
        avg_recall = total_recall / len(groups) if groups else 0
        self.energy_log.append(-avg_recall)
        return -avg_recall

    def _generate_neighbor(self, current_solution: list):
        """生成邻居解：随机交换或移动"""
        # 深拷贝
        new_group = deepcopy(current_solution)

        g1_idx, g2_idx = random.sample(range(len(new_group)), 2)
        group1 = new_group[g1_idx]
        group2 = new_group[g2_idx]

        prob = random.random()
        if prob < 0.3:
            self._swap(group1, group2)
        elif prob < 0.6:
            self._shift(group1, group2)
        elif prob < 0.8:
            new_group.remove(group1)
            new_group.remove(group2)
            new_group.extend(self._split(group1))
            new_group.extend(self._split(group2))
        elif prob < 0.98:
            if len(new_group) > 3:
                new_group.remove(group1)
                new_group.remove(group2)
                new_group.append(self._merge(group1, group2))
        else:
            if self.T > 4.0:
                new_group = self._recombine()

        new_group = [g for g in new_group if len(g) > 0]
        return new_group

    def _swap(self, group1: list, group2: list):
        d1_num = len(group1)
        d2_num = len(group2)
        if d1_num < self.min_size or d2_num < self.min_size:
            return group1, group2
        
        d1_idx, d2_idx = random.randint(0, d1_num-1), random.randint(0, d2_num-1)
        d1, d2 = group1[d1_idx], group2[d2_idx]
        if d1 != d2:
            group1.remove(d1)
            group2.remove(d2)
            group1.append(d2)
            group2.append(d1)
        return group1, group2
    
    def _shift(self, group1, group2):
        idx = random.randint(0, len(group1) - 1)
        d = group1[idx]
        if d not in group2:
            group1.remove(d)
            group2.append(d)
        return group1, group2
    
    def _split(self, group):
        if len(group) <= self.min_size:
            return [group]
        
        split_point = random.randint(1, len(group) -1)
        random.shuffle(group)
        g1 = group[: split_point]
        g2 = group[split_point:]
        return [g1, g2]
    
    def _merge(self, group1, group2):
        return group1 + group2

    def _recombine(self):
        new_group = []
        self._flattern()
        if self.estimator.elite_group:
            max_sample = min(len(self.estimator.elite_group), len(self.all_)//self.max_size)
            elite_group = random.sample(list(self.estimator.elite_group), random.randint(1, max_sample))
            if elite_group:
                for group in elite_group:
                    self.all_.difference_update(group)
                    new_group.append(list(group))
        if self.all_:
            tempoary = self.all_.union(self.estimator.poor_disease)
            while True:
                poor = random.sample(list(tempoary), random.randint(self.min_size, self.max_size))
                new_group.append(poor)
                for i in poor:
                    tempoary.remove(i)
                if self.max_size > len(tempoary):
                    break
            new_group.append(list(tempoary))
            self.estimator.poor_disease = set()
        return new_group
    
    def _flattern(self,):
        self.all_ = set()
        for group in self.insolution:
            for di in group:
                self.all_.add(di)

    def solve(self):
        """
        Return best_solution `List[DiseaseGroup]`, -best_energy `float`
        """
        current_solution = self.insolution.copy()
        current_energy = self._calculate_energy(current_solution)
        best_solution = current_solution
        best_energy = current_energy
        
        print(f"初始解 Recall: {current_energy:.4f}")
        no_improve = 0
        for i in range(self.iteration):
            self.T /= (1+0.01*i)
            if self.T < 1e-2: break

            # 随机选一个邻居
            neighbor_solution = self._generate_neighbor(current_solution)
            neighbor_energy = self._calculate_energy(neighbor_solution)
            
            # 接受概率 (Metropolis 准则)
            # 如果新解更好，或者以一定概率接受差解
            delta_e = neighbor_energy - current_energy
            if delta_e < 0 or random.random() < math.exp(-delta_e / self.T):
                current_solution = neighbor_solution
                current_energy = neighbor_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_solution = current_solution
                    no_improve = 0
                else:
                    no_improve += 1
                
                if self.T < 1.0 and no_improve > self.iteration//3:
                    print("提前收敛")
                    break

            if random.random() < 0.01:
                 print(f"当前温度: {self.T:.2f}, 当前最佳 Recall: {-best_energy:.4f}")

        return best_solution, -best_energy
