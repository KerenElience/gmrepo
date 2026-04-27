import pickle
import pandas as pd
from utils.common import BaseModule
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import classification_report

class RFModel(BaseModule):
    def __init__(self, params: dict = None):
        self.params = {
            "n_estimators": 100,
            "max_features": "sqrt",
            "min_samples_split": 0.2,
            "min_samples_leaf": 0.01,
            "oob_score": True,
            "n_jobs": -1,
            "class_weight": None,

            "random_state": 42
        }
        if params is not None:
            self.params.update(params)
        self.seed = self.params.get("random_state")
        self.rf = RandomForestClassifier(**self.params)

    def train(self, x, y):
        self.rf.fit(x, y)
        if self.params.get("oob_score"):
            print(f"OOB Score: {self.rf.oob_score_:.4f}")
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state = self.seed)
        scores = cross_val_score(self.rf, x, y, cv = cv, scoring="f1_macro", n_jobs=-1)
        return scores.mean()

    def eval(self, x, y, cls_name = None):
        y_pred = self.rf.predict(x)
        if cls_name is None:
            cls_name = list(set(y))
        cls_report = classification_report(y, y_pred, target_names = cls_name,
                                           output_dict = True)
        return y_pred, pd.DataFrame(cls_report).T
    
    def save(self, savepath):
        try:
            with open(savepath, "wb") as f:
                pickle.dump(self.rf, f)
            print("RandomForest Model saved.")
        except Exception as e:
            print(e)