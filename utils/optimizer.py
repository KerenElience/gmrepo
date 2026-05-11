import optuna
import json
from numpy.typing import NDArray
from optuna import Trial
from optuna.pruners import MedianPruner
from typing import Literal


from gmrepo.src.evaluator import Evaluator
from gmrepo.src.models.multi_disease_classifier import MLModel

class Optimizer():
    """
    Bayes optimizer for random forest
    """
    def __init__(self, model_type: str, x: NDArray, y: NDArray, n_trials: int = 50):
        self.model_type = model_type
        self.x = x
        self.y = y
        self.n_trials = n_trials
        self.seed = 42

        self.default_config, self.search_config = self._load_configs()

    def _load_conf(self, default_config: str, search_config: str):
        with open(default_config, "r") as f:
            default_params = json.load(f)

        with open(search_config, "r") as f:
            search_params = json.load(f)
        return default_params, search_params

    def objective(self, trial: Trial):
        """
        objective functional
        """
        ## 添加参数搜索范围
        params = self._suggest_params(trial)
        model = MLModel(self.model_type, params)
        criterion = Evaluator()
        model.train(self.x, self.y)
        score = criterion.cv(model, self.x, self.y)
        return score
    
    def run(self, direction: Literal["maximize", "minimize"] = "maximize"):
        print(f"[INFO] Starting search best parameters used bayes, about epoch {self.n_trials}")
        pruner = MedianPruner() if self.model_type == "xgb" else None
        
        study = optuna.create_study(
            direction=direction,
            sampler = optuna.samplers.TPESampler(seed = self.seed),
            pruner=pruner
        )

        study.optimize(self.objective, n_trials = self.n_trials, show_progress_bar = True)
        print(f"\n Optimized complete: best_value: {study.best_value:.4f}")
        return study.best_params

    def _suggest_params(self, trial: Trial):
        
        default_params = self.default_config[self.model_type]
        search_space = self.search_config[self.model_type]
        final_params = {}
        for k, v in default_params.items():
            if k not in search_space:
                final_params[k] = v
        
        for param, config in search_space.items():
            p_type = config["type"]
            p_value = config["value"]

            if p_type == "int":
                final_params[param] = trial.suggest_int(param, p_value[0], p_value[1])
            elif p_type == "float":
                log = config.get("log", False)
                final_params[param] = trial.suggest_float(param, p_value[0], p_value[1], log=log)
            elif p_type == "categorical":
                final_params[param] = trial.suggest_categorical(param, p_value)
            
        return final_params
