import pickle, json
from utils.common import BaseModule
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import recall_score, f1_score
from typing import Literal

class MLModel(BaseModule):
    def __init__(self, model_type: Literal["rf", "xgb"]):
        super().__init__()
        params = self._load_param()
        self.params = params[model_type] 
        self.seed = self.params.get("random_state")
        self.model = RandomForestClassifier(**self.params) if model_type == "rf" else XGBClassifier(*self.params)

    def train(self, x, y):
        self.model.fit(x, y)
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state = self.seed)
        scores = cross_val_score(self.rf, x, y, cv = cv, scoring="f1_macro", n_jobs=-1)
        return scores.mean()

    def eval(self, x, y):
        """
        Return recall, f1_score
        """
        y_pred = self.model.predict(x)
        if cls_name is None:
            cls_name = list(set(y))
        recall = recall_score(y, y_pred)
        try:
            f1 = f1_score(y, y_pred, average="macro")
        except:
            f1 = None
        return recall, f1
    
    def _load_param(self, configpath: str = "../../config/default_config.json" ):
        with open(configpath, "r") as f:
            params = json.load(f)
        return params

    def save(self, savepath):
        try:
            with open(savepath, "wb") as f:
                pickle.dump(self.model, f)
            print("Model saved.")
        except Exception as e:
            print(e)