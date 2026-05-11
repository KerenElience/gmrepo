import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import recall_score, f1_score
from gmrepo.src.models.multi_disease_classifier import MLModel
from gmrepo.utils.process import DataProcess
from utils.utils import calculate_group_hash, GroupCache
# from gmrepo.src.utils import upsample

class Evaluator():
    def __init__(self, seed=42, n_splits=4):
        self.seed = seed
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

    def cv(self, model, x, y, scoring="f1_macro"):
        scores = cross_val_score(model, x, y, cv=self.cv, scoring=scoring, n_jobs=-1)
        return scores.mean()

    def get_metrics(self, model, x_ture, y_true):
        y_pred = model.eval(x_ture)
        recall = recall_score(y_true, y_pred)
        try:
            f1 = f1_score(y_true, y_pred, average="macro")
        except:
            f1 = None
        return recall, f1
    
class DIestimator():
    def __init__(self, model: MLModel, prcd: DataProcess, min_size: int = 2, max_size: int = 5):
        self.prcd = prcd
        self.cache = GroupCache()
        self.model = model
        self.evaluator = Evaluator()

        self.min_size = min_size
        self.max_size = max_size
        self.elite_group = set()
        self.elite_disease = set()
        self.poor_disease = set()

    def get_metrics(self, diseases: list):
        penalty = 0.0
        hash = calculate_group_hash(diseases)
        x, y = self.prcd.get_sub_data(diseases)
        cls_name = self.prcd.encoder.classes_
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42, shuffle=True, stratify = y)
        ## SMOTE
        # _, counts = np.unique_counts(y)
        # x_train, x_test, y_train = upsample(x_train, x_test, y_train)
        self.model.train(x_train, y_train)
        recall, f1_score = self.evaluator.get_metrics(self.model, x_test, y_test)
        mean_recall = np.mean(recall)
        std_recall = np.std(recall)
        if len(diseases) < self.min_size:
            penalty += 0.8*(self.min_size - len(diseases)+ 1)
        elif len(diseases) > self.max_size:
            penalty += 0.8*(len(diseases) - self.max_size + 1)
        if f1_score is None:
            std_recall = penalty
        # (f1+mean_recall)/2: [0, 1]; std_recall: [0, 1); penalty: [0, 0.8*len(groups)]
        score = (f1_score + mean_recall)/2 - std_recall - penalty   #[-0.8*len(groups), 1]

        ## update
        if score > 0.90:
            self.elite_group.add(tuple(sorted(diseases)))
            self.elite_disease.update(diseases)
        zero_recall = [cls_name[i] for i in np.where(recall < 0.3)]
        for i in zero_recall:
            if i not in self.elite_disease:
                self.poor_disease.add(i)

        self.cache.cache[hash] = {"diseases": cls_name, "recall": recall, "f1_score": f1_score}
        return score
    
    