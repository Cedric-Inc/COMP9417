import numpy as np
import xgboost as xgb
from sklearn.utils.class_weight import compute_class_weight


def compute_effective_number_weights(y, beta=0.9999):
    y = np.array(y)
    classes, counts = np.unique(y, return_counts=True)

    effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    class_weights = 1.0 / effective_num
    class_weights = class_weights / np.sum(class_weights) * len(classes)

    class_weight_dict = dict(zip(classes, class_weights))
    sample_weight = np.array([class_weight_dict[label] for label in y])
    return sample_weight, class_weight_dict


def default_XGBoost(X_train, y_train, class_weight='balanced', train=True):
    classes = np.unique(y_train)

    if class_weight=='balanced':
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = dict(zip(classes, class_weights))
    elif class_weight=='effective':
        sample_weight, class_weight_dict = compute_effective_number_weights(y_train)

    sample_weights = np.array([class_weight_dict[label] for label in y_train])

    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(classes),
        eval_metric='mlogloss',
        tree_method='hist',
        device='cuda',
        max_depth=5,
        learning_rate=0.1051,
        n_estimators=100,
        subsample=0.6624,
        colsample_bytree=0.7498,
        random_state=42,
        n_jobs=-1,
        min_child_weight=5
    )
    if train:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    return model



