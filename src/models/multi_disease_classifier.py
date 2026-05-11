import pickle
from utils.common import BaseModule
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from typing import Literal

class MLModel(BaseModule):
    def __init__(self, model_type: Literal["rf", "xgb"], params: dict):
        super().__init__()
        self.model_type = model_type
        self.params = params
        self.seed = self.params.get("random_state")
        self.model = RandomForestClassifier(**self.params) if model_type == "rf" else XGBClassifier(**self.default_params)

    def train(self, x, y):
        self.model.fit(x, y)

    def eval(self, x):
        y_pred = self.model.predict(x)
        return y_pred

    def save(self, savepath):
        try:
            with open(savepath, "wb") as f:
                pickle.dump(self.model, f)
            print("Model saved.")
        except Exception as e:
            print(e)