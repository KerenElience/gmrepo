import numpy as np
import random
import math
from pydantic import BaseModel
from typing import Union
from randomforest import RFModel, classification_report, pd
from utils.process import calc_dist_matrix

class DiseaseGroup(BaseModel):
    disease: list
    x: np.ndarray
    y: Union[np.ndarray, pd.Series, list]
    model: Union[RFModel, None]

class SimulatedAnnealing:
    def __init__(self, x, y, disease_names, initial_temp=100.0, cooling_rate=0.95):
        self.x = x
        self.y = y
        self.names = disease_names
        self.n = len(disease_names)
        self.T = initial_temp
        self.cooling_rate = cooling_rate
        
        # 预计算掩码，加速数据提取
        self.data_cache = {}

    def _get_recall(self, group_indices):
        """快速计算一组的 Recall"""
        group_key = tuple(sorted(group_indices))
        if group_key in self.data_cache:
            X_g, y_g = self.data_cache[group_key]
        else:
            names = [self.names[i] for i in group_indices]
            mask = np.isin(self.y, names)
            X_g, y_g = self.X[mask], self.y[mask]
            self.data_cache[group_key] = (X_g, y_g)
            
        if len(np.unique(y_g)) < 2: return 0.0
        
        clf = RFModel()
        _ = clf.train()
        # 这里的 scoring 使用 recall_macro
        y_pred, metrics = RFModel.eval()
        return metrics

    def _calculate_energy(self, individual):
        """计算总能量 (即总 Recall)"""
        total_recall = 0
        valid = True
        
        for group in individual:
            if not (2 <= len(group) <= 5):
                return -1000 # 惩罚
            
            r = self._get_recall(group)
            total_recall += r
            
        return total_recall / len(individual)

    def _generate_neighbor(self, current_solution):
        """生成邻居解：随机交换或移动"""
        # 深拷贝
        new_solution = [g[:] for g in current_solution]
        
        # 策略1: 随机交换两个疾病的组别
        if random.random() < 0.5:
            g1_idx, g2_idx = random.sample(range(len(new_solution)), 2)
            if new_solution[g1_idx] and new_solution[g2_idx]:
                # 交换元素
                i1 = random.randint(0, len(new_solution[g1_idx]) - 1)
                i2 = random.randint(0, len(new_solution[g2_idx]) - 1)
                new_solution[g1_idx][i1], new_solution[g2_idx][i2] = \
                new_solution[g2_idx][i2], new_solution[g1_idx][i1]
        
        # 策略2: 随机把一个病移到另一个组
        else:
            g_from_idx = random.randint(0, len(new_solution) - 1)
            if len(new_solution[g_from_idx]) > 2: # 移出后不能少于2个
                disease = new_solution[g_from_idx].pop(random.randint(0, len(new_solution[g_from_idx]) - 1))
                
                g_to_idx = random.randint(0, len(new_solution) - 1)
                if len(new_solution[g_to_idx]) < 5: # 移入后不能多于5个
                    new_solution[g_to_idx].append(disease)
                else:
                    # 如果目标组满了，放回原处或新建一组
                    new_solution[g_from_idx].append(disease)
                    
        # 清理空组
        new_solution = [g for g in new_solution if g]
        return new_solution

    def solve(self, initial_solution):
        current_solution = initial_solution
        current_energy = self._calculate_energy(current_solution)
        best_solution = current_solution
        best_energy = current_energy
        
        print(f"初始解 Recall: {current_energy:.4f}")
        
        while self.T > 1.0:
            # 随机选一个邻居
            neighbor_solution = self._generate_neighbor(current_solution)
            neighbor_energy = self._calculate_energy(neighbor_solution)
            
            # 接受概率 (Metropolis 准则)
            # 如果新解更好，或者以一定概率接受差解
            delta_e = neighbor_energy - current_energy
            if delta_e > 0 or random.random() < math.exp(delta_e / self.T):
                current_solution = neighbor_solution
                current_energy = neighbor_energy
                
                if current_energy > best_energy:
                    best_energy = current_energy
                    best_solution = current_solution
            
            self.T *= self.cooling_rate
            # 打印进度
            if random.random() < 0.01:
                 print(f"当前温度: {self.T:.2f}, 当前最佳 Recall: {best_energy:.4f}")

        return best_solution, best_energy


def get_initial_guess(dist_matrix: np.ndarray, disease_names: list, min_partners: int = 2, max_partners: int=3):
    """
    Traverse all diseases as cores to find the optimal combination of 'negative samples' for each core

    - dist_matrix: distance matrix for echo elements.
    - disease_names: disease name, code or phenotype.
    - min_partners: Include at least a few negative samples in the combination (avoid combinations that are too simple)
    - max_partners: Up to a few negative samples in the assembly (controlling for model complexity)
    """
    results = []
    n = len(disease_names)
    grouped = []
    
    for i in range(n):
        core_name = disease_names[i]
        if core_name in grouped:
            continue
        grouped.append(core_name)
        # 获取当前核心疾病到其他所有疾病的距离
        distances = dist_matrix[i]
        each_dist = []
        
        # 排除自己到自己的距离 (通常为0)
        other_distances = []
        other_indices = []
        
        for j in range(n):
            if i != j:
                if disease_names[j] in grouped:
                    continue
                other_distances.append(distances[j])
                other_indices.append(j)
        
        paired = list(zip(other_indices, other_distances))
        paired.sort(key=lambda x: x[1], reverse=True)
        
        selected_count = min(max_partners, len(paired))
        
        # 确保至少有 min_partners 个伙伴，否则这个组合可能太简单或没意义
        if selected_count >= min_partners:
            partners = []
            total_distance_score = 0
            
            for idx, dist in paired[:selected_count]:
                if disease_names[idx] in grouped:
                    continue
                partners.append(disease_names[idx])
                grouped.append(disease_names[idx])
                total_distance_score += dist
            
            # 计算平均距离作为"组合区分度得分"
            avg_score = total_distance_score / selected_count
            
            results.append({
                "核心疾病": core_name,
                "组合伙伴": partners,
                "伙伴数量": len(partners),
                "平均区分度得分": avg_score
            })
        else:
            # 如果连 min_partners 个都凑不齐（比如总共就3个病），则特殊处理
            partners = [disease_names[idx] for idx, _ in paired]
            avg_score = np.mean([d for _, d in paired])
            results.append({
                "核心疾病": core_name,
                "组合伙伴": partners,
                "伙伴数量": len(partners),
                "平均区分度得分": avg_score
            })

    # 将结果转换为 DataFrame 方便查看和排序
    df_results = pd.DataFrame(results)
    
    # 按"平均区分度得分"降序排列
    # 得分越高，说明核心疾病和伙伴疾病分得越开，模型越好训练
    df_results.sort_values(by="平均区分度得分", ascending=False, inplace=True)
    df_results.reset_index(drop=True, inplace=True)

    return df_results