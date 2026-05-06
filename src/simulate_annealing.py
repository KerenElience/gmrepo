import numpy as np
import random
import math
from numpy.typing import NDArray
from typing import Optional, Union
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from .randomforest import RFModel, pd
from utils.utils import calculate_group_hash
from imblearn.over_sampling import SMOTE, ADASYN

class GroupCache:
    cache = {}

    @classmethod
    def clear(cls):
        cls.cache.clear()

class DIestimator():
    def __init__(self, disease: list,
                 label: Union[pd.Series, NDArray],
                 x: NDArray,
                 encoder: LabelEncoder = None,
                 model: Optional["RFModel"] = None):
        self.disease = disease
        self.label = label
        self.x = x
        self.encoder = encoder if encoder is not None else LabelEncoder()
        self.model = model
        self.cache = GroupCache()
        self.y = self.encoder.fit_transform(label)

    def get_current_data(self, mask_disease: list):
        mask = self.label.isin(mask_disease)
        mask_label = self.label[mask]
        x_group = self.x[mask]
        y_group = self.encoder.fit_transform(mask_label)
        return x_group, y_group

    def get_metrics(self, diseases: list = None):
        hash = calculate_group_hash(diseases)
        if hash in self.cache.cache:
            return self.cache.cache[hash]["metrics"]
        else:
            if self.model is None:
                self.model = RFModel()
            if diseases:
                x, y = self.get_current_data(diseases)
            else:
                x, y = self.x, self.y
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify = y)
            ## SMOTE
            _, counts = np.unique_counts(y)
            if counts.max()/counts.min() > 3.0:
                x_train, x_test, y_train = self._upsample(x_train, x_test, y_train)
            _ = self.model.train(x_train, y_train)
            _, metrics = self.model.eval(x_test, y_test, self.encoder.classes_)
            self.cache.cache[hash] = {"diseases": diseases, "metrics": metrics}
            return metrics

    def _upsample(self, x_train, x_test, y_train):
        scale = StandardScaler()
        x_train_scale = scale.fit_transform(x_train)
        xv_scale = scale.transform(x_test)
        smote = SMOTE(random_state=42, k_neighbors=7)
        x_smote, y_smote = smote.fit_resample(x_train_scale, y_train)
        return x_smote, xv_scale, y_smote

    def merge(self, diseases):
        for di in diseases:
            if di not in self.disease:
                raise ValueError(f"Input {di} not in group diseases")
        mask_idx = self.label.isin(diseases)
        return mask_idx

    def __len__(self):
        return len(self.disease)

class SimulatedAnnealing():
    def __init__(self, insolution: list, min_size: int = 2, 
                 max_size: int = 5, estimator: DIestimator = None,
                 initial_temp: float = 5.0,
                 cooling_rate: float = 0.95,
                 iteration: int = 100):
        self.insolution = insolution
        self.T = initial_temp
        self.cooling_rate = cooling_rate
        self.min_size = min_size
        self.max_size = max_size
        self.iteration = iteration
        self.energy_log = []

        self.elite_group = set()
        self.elite_disease = set()
        self.poor_disease = set()

        self.estimator = estimator

    def _get_recall(self, group: list):
        """快速计算一组的 Recall"""
        plenty = 0.0
        if len(group) < self.min_size:
            plenty += 0.8
        elif len(group) > self.max_size:
            plenty += 0.8*(len(group) - self.max_size)
        metrics = self.estimator.get_metrics(group)
        f1_score = metrics["f1-score"]["macro avg"]
        metrics = metrics.iloc[:-3, :]
        mean_recall = metrics["recall"].mean()
        std_recall = metrics["recall"].std()
        if f1_score == 1.0:
            std_recall = plenty
        score = f1_score + mean_recall - std_recall - plenty

        ## update
        if score > 1.7:
            self.elite_group.add(tuple(sorted(group)))
            self.elite_disease.update(group)
        zero_recall = metrics[metrics["recall"] < 0.3].index.to_list()
        for i in zero_recall:
            if i not in self.elite_disease:
                self.poor_disease.add(i)
        return score

    def _calculate_energy(self, groups: list):
        """计算总能量 (即总 Recall)"""
        total_recall = 0
        
        for group in groups:
            r = self._get_recall(group)
            total_recall += r
        avg_recall = total_recall / len(groups) if groups else 0
        self.energy_log.append(-avg_recall)
        return -avg_recall

    def _generate_neighbor(self, current_solution: list):
        """生成邻居解：随机交换或移动"""
        # 深拷贝
        new_group = current_solution.copy()

        g1_idx, g2_idx = random.sample(range(len(new_group)), 2)
        group1 = new_group[g1_idx]
        group2 = new_group[g2_idx]

        prob = random.random()
        if prob < 0.45:
            self._swap(group1, group2)
        elif prob < 0.9:
            self._shift(group1, group2)
        else:
            new_group = self._recombine()
        
        new_group = [g for g in new_group if len(g)>0]
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
        prob = random.random()
        if prob > 0.5:
            idx = random.randint(0, len(group1) - 1)
            d = group1[idx]
            if d not in group2:
                group1.remove(d)
                group2.append(d)
        else:
            idx = random.randint(0, len(group2) - 1)
            d = group2[idx]
            if d not in group1:
                group2.remove(d)
                group1.append(d)
        return group1, group2
    
    def _recombine(self):
        new_group = []
        self._flattern()
        max_sample = min(len(self.elite_group), len(self.all_)//self.max_size)
        elite_group = random.sample(list(self.elite_group), random.randint(1, max_sample))
        if elite_group:
            for group in elite_group:
                self.all_.difference_update(group)
                new_group.append(list(group))
        if self.all_:
            tempoary = self.all_.union(self.poor_disease)
            while True:
                poor = random.sample(list(tempoary), random.randint(self.min_size, self.max_size))
                new_group.append(poor)
                for i in poor:
                    tempoary.remove(i)
                if self.max_size > len(tempoary):
                    break
            new_group.append(list(tempoary))
            self.poor_disease = set()
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
        
        for i in range(self.iteration):
            self.T *= self.cooling_rate
            if self.T < 1e-2: break

            # 随机选一个邻居
            neighbor_solution = self._generate_neighbor(current_solution)
            neighbor_energy = self._calculate_energy(neighbor_solution)
            
            # 接受概率 (Metropolis 准则)
            # 如果新解更好，或者以一定概率接受差解
            delta_e = neighbor_energy - current_energy
            if delta_e > 0 or random.random() < math.exp(-delta_e / self.T):
                current_solution = neighbor_solution
                current_energy = neighbor_energy
                
                if current_energy > best_energy:
                    best_energy = current_energy
                    best_solution = current_solution.copy()

            if random.random() < 0.01:
                 print(f"当前温度: {self.T:.2f}, 当前最佳 Recall: {best_energy:.4f}")

        return best_solution, -best_energy


def get_initial_guess(dist_matrix: np.ndarray, disease_names: list, 
                      min_partners: int = 1, max_partners: int=1):
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
    if min_partners > max_partners:
        raise ValueError(f"Input min partners more than max partner.")
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
            partners.append(core_name)
            results.append({
                "group": sorted(partners),
                "mean_dist": mean_dist
            })
        else:
            # 如果连 min_partners 个都凑不齐（比如总共就3个病），则特殊处理
            partners = [disease_names[idx] for idx, _ in paired]
            mean_dist = np.mean([d for _, d in paired])
            partners.append(core_name)
            results.append({
                "group": sorted(partners),
                "mean_dist": mean_dist
            })

    df_results = pd.DataFrame(results)
    df_results.sort_values(by="mean_dist", ascending=False, inplace=True)
    df_results.reset_index(drop=True, inplace=True)

    return df_results