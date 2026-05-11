import json
from gmrepo.src.models.multi_disease_classifier import MLModel
from gmrepo.utils.optimizer import Optimizer
from gmrepo.utils.process import DataProcess
from gmrepo.utils.utils import calculate_group_hash

def moe_train(meta: str, model_type: str, solution: list, param_savepath: str):
    prcd = DataProcess(meta)
    data = prcd.filtration(data)
    x = prcd.clr_transform(prcd)
    params = dict()
    for group in solution:
        x_sub, y_sub = prcd.get_sub_data(group)
        moe_optim = Optimizer(model_type, x_sub, y_sub)
        name = calculate_group_hash(group)
        try:
            best_param = moe_optim.run()
            params[name] = best_param
        except Exception as e:
            print(f"Optuna {group} occur error: {e}")
    with open(param_savepath, "r", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return None