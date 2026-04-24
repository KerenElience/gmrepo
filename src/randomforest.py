import pickle
import optuna
import pandas as pd
from optuna import Trial
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

class RFModel():
    def __init__(self, x_train, x_test, y_train, y_test, cls_name, **kwargs):
        self.params = {
            "n_estimators": 300,
            "max_features": "sqrt",
            "min_samples_split": 0.2,
            "min_samples_leaf": 0.01,
            "oob_score": True,
            "n_jobs": -1,
            "class_weight": None,

            "random_state": 42
        }
        self.params.update(kwargs)
        self.seed = self.params.get("random_state")
        self.x_train, self.x_test, self.y_train, self.y_test = x_train, x_test, y_train, y_test
        self.cls_name = cls_name
        self.rf = RandomForestClassifier(**self.params)

    def train(self):
        self.rf.fit(self.x_train, self.y_train)
        if self.params.get("oob_score"):
            print(f"OOB Score: {self.rf.oob_score_:.4f}")

        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state = self.seed)
        scores = cross_val_score(self.rf, self.x_train, self.y_train, cv = cv, scoring="f1_macro", n_jobs=-1)
        return scores.mean()

    def eval(self,):
        y_pred = self.rf.predict(self.x_test)
        cls_report = classification_report(self.y_test, y_pred, target_names = self.cls_name,
                                            output_dict = True)
        return y_pred, pd.DataFrame(cls_report).T
    
    def save(self, savepath):
        try:
            with open(savepath, "wb") as f:
                pickle.dump(self.rf, f)
            print("RandomForest Model saved.")
        except Exception as e:
            print(e)

    @property
    def get_cm(y_test, y_pred):
        cm = confusion_matrix(y_test, y_pred)
        return cm

class RFOptimizer():
    """
    Bayes optimizer for random forest
    """
    def __init__(self, x, y, n_trials = 50):
        self.x = x
        self.y = y
        self.n_trials = n_trials
        self.seed = 42
        self.cv = StratifiedKFold(n_splits=4, shuffle=True, random_state = self.seed)

    def objective(self, trial:Trial):
        """
        objective functional
        """
        params = {
            "n_estimators":trial.suggest_int("n_estimators", 50, 500),
            "max_depth":trial.suggest_int("max_depth", 3, 12), ## features 99< 2^7
            "min_samples_split":trial.suggest_float("min_samples_split", 0.01, 0.3),
            "min_samples_leaf":trial.suggest_float("min_samples_leaf", 0.01, 0.1),

            "max_features":trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }
        rf = RFModel(self.x, None, self.y, None, None, **params,)
        scores = rf.train()
        return scores
    
    def run(self):
        print(f"[INFO] Starting search best parameters used bayes, about epoch {self.n_trials}")
        study = optuna.create_study(
            direction="maximize",
            sampler = optuna.samplers.TPESampler(seed = self.seed)
        )

        study.optimize(self.objective, n_trials = self.n_trials, show_progress_bar = True)
        print(f"\n Optimized complete: best_value: {study.best_value:.4f}")
        return study.best_params