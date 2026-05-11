import random
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

def random_partition(diseases: list, min_size=2, max_size=3):
    diseases = diseases.copy()
    random.shuffle(diseases)
    groups = []
    i = 0
    n = len(diseases)

    while i < n:
        size = random.randint(min_size, max_size)
        group = diseases[i:i+size]
        groups.append(group)
        i += size
    return groups

def encode_solution(groups):
    return tuple(sorted(tuple(sorted(g)) for g in groups))

def upsample(x_train, x_test, y_train):
    scale = StandardScaler()
    x_train_scale = scale.fit_transform(x_train)
    xv_scale = scale.transform(x_test)
    smote = SMOTE(random_state=42, k_neighbors=7)
    x_smote, y_smote = smote.fit_resample(x_train_scale, y_train)
    return x_smote, xv_scale, y_smote

def platt_scale(probs, labels):
    "Calibrate the probs"
    calibrator = LogisticRegression()
    calibrator.fit(probs.reshape(-1,1), labels)

    calibrated_prob = calibrator.predict_proba(probs.reshape(-1,1))[:,1]
    return calibrated_prob