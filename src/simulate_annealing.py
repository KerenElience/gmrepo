import numpy as np
import random
import math
from numpy.typing import NDArray
from typing import Optional, Union, List, Any
from sklearn.model_selection import train_test_split
from .randomforest import RFModel, pd
from sklearn.base import clone

class DiseaseGroup():
    def __init__(self, disease: list,
                 label: Union[pd.Series, NDArray],
                 x: NDArray,
                 encoder: Any,
                 model: Optional["RFModel"] = None):
        self.disease = disease
        self.label = label
        self.x = x
        self.encoder = encoder
        self.model = model

        self.y = self.encoder.fit_transform(self.label)

    def get_metrics(self):
        if self.model is None:
            self.model = RFModel()
        x_train, x_test, y_train, y_test = train_test_split(self.x, self.y, test_size=0.2, shuffle=True, stratify = self.y)
        _ = self.model.train(x_train, y_train)
        _, metrics = self.model.eval(x_test, y_test)
        return metrics

    def pop(self, disease):
        """
        Return mask_label, x
        """
        mask_label = self.label.eq(disease)
        pop_label = self.label[mask_label]
        pop_x = self.x[mask_label]
        
        # update
        self.label = self.label[~mask_label]
        self.x = self.x[~mask_label]
        self.y = self.encoder.fit_transform(self.label)
        self.disease.remove(disease)
        return pop_label, pop_x
    
    def add(self, disease, label, x):
        if disease not in self.disease:
            self.disease.append(disease)
            self.label = pd.concat([self.label, label])
            self.x = np.vstack((self.x, x))
            self.y = self.encoder.fit_transform(self.label)

    def merge(self, disease_labels):
        for di in disease_labels:
            if di not in self.disease:
                raise ValueError(f"Input {di} not in group diseases")
        mask_idx = self.label.isin(disease_labels)
        # self.y = LabelEncoder.fit_transform(mask_idx)
        return mask_idx
    
    def copy(self):
        disease = self.disease.copy()
        label = self.label.copy()
        x = self.x.copy()
        encoder = clone(self.encoder)
        model = None
        return DiseaseGroup(disease, label, x, encoder, model)
    
    def __len__(self):
        return len(self.disease)

class SimulatedAnnealing():
    def __init__(self, insolution: list, min_size: int =2, 
                 max_size: int = 5, 
                 initial_temp: float = 100.0, 
                 cooling_rate: float = 0.95,
                 iteration: int = 50):
        self.insolution = insolution
        self.T = initial_temp
        self.cooling_rate = cooling_rate
        self.min_size = min_size
        self.max_size = max_size
        self.iteration = iteration
        self.energys = []

    def _get_recall(self, group: DiseaseGroup):
        """快速计算一组的 Recall"""
        if np.unique(len(group.disease)) < 2: return 0.0
        metrics = group.get_metrics()
        f1_score = metrics["f1-score"]["macro avg"]
        return f1_score

    def _calculate_energy(self, groups: List[DiseaseGroup]):
        """计算总能量 (即总 Recall)"""
        total_recall = 0
        
        for group in groups:
            if not (self.min_size <= len(group) <= self.max_size):
                return 100 # 惩罚
            
            r = self._get_recall(group)
            total_recall += r
        avg_recall = total_recall / len(groups) if groups else 0
        self.energys.append(-avg_recall)
        return -avg_recall

    def _generate_neighbor(self, current_solution: List[DiseaseGroup]):
        """生成邻居解：随机交换或移动"""
        # 深拷贝
        new_group = [g.copy() for g in current_solution]

        g1_idx, g2_idx = random.sample(range(len(new_group)), 2)
        group1 = new_group[g1_idx]
        group2 = new_group[g2_idx]
        if random.random() < 0.6:
            group1, group2 = self._swap(group1, group2)
        else:
            group1, group2 = self._shift(group1, group2)
        new_group[g1_idx] = group1
        new_group[g2_idx] = group2
        new_group = [g for g in new_group if len(g)>0]
        return new_group

    def _swap(self, group1, group2):
        d1_num = len(group1)
        d2_num = len(group2)
        if d1_num < self.min_size or d2_num < self.min_size:
            return group1, group2
        
        d1_idx, d2_idx = random.randint(0, d1_num-1), random.randint(0, d2_num-1)
        d1, d2 = group1.disease[d1_idx], group2.disease[d2_idx]
        d1_mask, d1_x = group1.pop(d1)
        d2_mask, d2_x = group2.pop(d2)

        group1.add(d2, d2_mask, d2_x)
        group2.add(d1, d1_mask, d1_x)
        if group1.x.shape[0] != group1.label.shape[0]:
            raise ValueError(f"组中{d1}交换{d2}失败，维度不一致")
        return group1, group2
    
    def _shift(self, group1, group2):
        prob = random.random()
        if prob > 0.5:
            idx = random.randint(0, len(group1) - 1)
            print(f'取值{idx}, 组长度{len(group1)}')
            d = group1.disease[idx]
            d_mask, d_x = group1.pop(d)
            group2.add(d, d_mask, d_x)
        else:
            idx = random.randint(0, len(group2) - 1)
            print(f'取值{idx}, 组长度{len(group2)}')
            d = group2.disease[idx]
            d_mask, d_x = group2.pop(d)
            group1.add(d, d_mask, d_x)
        return group1, group2
    
    def _split_and_regroup(self, groups: List[DiseaseGroup]):
        groups_scores = []
        for i, g in enumerate(groups):
            df = g.get_metrics()
            df.sort_values("recall", ascending=False, inplace=True)
        victims = []
        pool_disease = []
        num_victims = random.randomint(1, )
        pass

    def solve(self):
        """
        Return best_solution `List[DiseaseGroup]`, -best_energy `float`
        """
        current_solution = self.insolution.copy()
        current_energy = self._calculate_energy(current_solution)
        best_solution = current_solution
        best_energy = current_energy
        
        print(f"初始解 Recall: {current_energy:.4f}")
        
        for i in range(self.iteration):
            self.T *= self.cooling_rate
            if self.T < 1e-3: break

            # 随机选一个邻居
            neighbor_solution = self._generate_neighbor(current_solution)
            neighbor_energy = self._calculate_energy(neighbor_solution)
            
            # 接受概率 (Metropolis 准则)
            # 如果新解更好，或者以一定概率接受差解
            delta_e = neighbor_energy - current_energy
            if delta_e < 0 or random.random() < math.exp(-delta_e / self.T):
                current_solution = neighbor_solution
                current_energy = neighbor_energy
                
                if current_energy > best_energy:
                    best_energy = current_energy
                    best_solution = current_solution

            if random.random() < 0.01:
                 print(f"当前温度: {self.T:.2f}, 当前最佳 Recall: {best_energy:.4f}")

        return best_solution, -best_energy


def get_initial_guess(dist_matrix: np.ndarray, disease_names: list, min_partners: int = 1, max_partners: int=2):
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
            
            mean_dist = total_distance_score / selected_count
            
            results.append({
                "core": core_name,
                "partners": partners,
                "num_partner": len(partners),
                "mean_dist": mean_dist
            })
        else:
            # 如果连 min_partners 个都凑不齐（比如总共就3个病），则特殊处理
            partners = [disease_names[idx] for idx, _ in paired]
            mean_dist = np.mean([d for _, d in paired])
            results.append({
                "core": core_name,
                "partners": partners,
                "num_partner": len(partners),
                "mean_dist": mean_dist
            })

    df_results = pd.DataFrame(results)
    df_results.sort_values(by="mean_dist", ascending=False, inplace=True)
    df_results.reset_index(drop=True, inplace=True)

    return df_results

def get_insolution(x, encoder, label: pd.Series, df: pd.DataFrame):
    """
    x: np.ndarray
    encoder: LabelEncoder
    label: disease/phenotype str
    df: insolution about disease group by distance matrix.
    """
    insolution = []
    for core, partneres in df[["core", "partners"]].values:
        disease = partneres
        disease.append(core)
        mask = label.isin(disease)
        mask_x = x[mask]
        mask_label = label[mask]
        group = DiseaseGroup(disease, mask_label, mask_x, encoder, None)
        insolution.append(group)
    return insolution