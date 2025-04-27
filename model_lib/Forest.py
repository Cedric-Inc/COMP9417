from sklearn.ensemble import RandomForestClassifier
import numpy as np

def compute_effective_number_weights(y, beta=0.9999):
    y = np.array(y)
    classes, counts = np.unique(y, return_counts=True)

    effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    class_weights = 1.0 / effective_num
    class_weights = class_weights / np.sum(class_weights) * len(classes)

    class_weight_dict = dict(zip(classes, class_weights))
    sample_weight = np.array([class_weight_dict[label] for label in y])
    return sample_weight, class_weight_dict

def default_random_forest(X_train, y_train, class_weight="effective", train=True):

    y_train = np.array(y_train).reshape(-1)
    if class_weight == "effective":
        sample_weight, class_weight_dict = compute_effective_number_weights(y_train)
    else:
        class_weight_dict = "balanced_subsample"

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=20,
        max_features='sqrt',
        class_weight=class_weight_dict,
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    if train:
        model.fit(X_train, y_train)
    return model
