import numpy as np
from sklearn.model_selection import train_test_split
from gmrepo.src.models.multi_disease_classifier import MLModel
from gmrepo.utils.process import DataProcess
from utils.utils import calculate_group_hash, GroupCache

# from gmrepo.src.utils import upsample

class DIestimator():
    def __init__(self, prcd: DataProcess):
        self.prcd = prcd
        self.cache = GroupCache()
        self.model = MLModel("rf")

    def get_metrics(self, diseases: list):
        hash = calculate_group_hash(diseases)
        x, y = self.prcd.get_sub_data(diseases)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True, stratify = y)
        ## SMOTE
        # _, counts = np.unique_counts(y)
        # if counts.max()/counts.min() > 3.0:
        #     x_train, x_test, y_train = upsample(x_train, x_test, y_train)
        _ = self.model.train(x_train, y_train)
        recall, f1_score = self.model.eval(x_test, y_test)
        self.cache.cache[hash] = {"diseases": diseases, "recall": recall, "f1_score": f1_score}
        return recall, f1_score