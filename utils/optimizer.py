import optuna
from numpy.typing import NDArray
from optuna import Trial
from typing import Literal
from sklearn.model_selection import StratifiedKFold
from .common import BaseModule

class Optimizer():
    """
    Bayes optimizer for random forest
    """
    def __init__(self, model: BaseModule, model_parameters: dict, x: NDArray, y: NDArray, n_trials: int = 50):
        self.model = model
        self.model_parameters = model_parameters
        self.x = x
        self.y = y
        self.n_trials = n_trials
        self.seed = 42
        self.cv = StratifiedKFold(n_splits=4, shuffle=True, random_state = self.seed)

    def objective(self, trial:Trial):
        """
        objective functional
        """
        if self.model_parameters is None:
            self.model_parameters = {
                "n_estimators":trial.suggest_int("n_estimators", 50, 500),
                "max_depth":trial.suggest_int("max_depth", 3, 12), ## features 99< 2^7
                "min_samples_split":trial.suggest_float("min_samples_split", 0.01, 0.3),
                "min_samples_leaf":trial.suggest_float("min_samples_leaf", 0.01, 0.1),

                "max_features":trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            }
        scores = self.model.train(self.x, self.y)
        return scores
    
    def run(self, direction: Literal["maximize", "minimize"] = "maximize"):
        print(f"[INFO] Starting search best parameters used bayes, about epoch {self.n_trials}")
        study = optuna.create_study(
            direction=direction,
            sampler = optuna.samplers.TPESampler(seed = self.seed),
        )

        study.optimize(self.objective, n_trials = self.n_trials, show_progress_bar = True)
        print(f"\n Optimized complete: best_value: {study.best_value:.4f}")
        return study.best_params