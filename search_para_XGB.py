from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from solving_imbalance.Sampling import data_sampler


def compute_effective_number_weights(y, beta=0.9999):
    y = np.array(y)
    classes, counts = np.unique(y, return_counts=True)

    effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    class_weights = 1.0 / effective_num
    class_weights = class_weights / np.sum(class_weights) * len(classes)

    class_weight_dict = dict(zip(classes, class_weights))
    return class_weight_dict


def train():

    X = pd.read_csv('Data/X_train.csv')
    y = pd.read_csv('Data/y_train.csv')
    y = np.array(y).reshape(-1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, y_train = data_sampler(X_train, y_train, oversampling_methods=['SMOTE', 'ADASYN'],
                                    undersampling_methods=['TOMEKLINKS'], minority_samples=100,
                                    majority_samples=400,use_pca=False, use_scaling=True)

    from xgboost import XGBClassifier
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.utils.class_weight import compute_class_weight
    from scipy.stats import randint, uniform

    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))
    sample_weight = np.array([class_weight_dict[i] for i in y_train])

    xgb_model = XGBClassifier(
        objective='multi:softprob',  # or 'binary:logistic'
        eval_metric='mlogloss',
        tree_method='hist',
        device='cuda',
        random_state=42
    )

    param_dist = {
        'max_depth': randint(3, 10),
        'min_child_weight': randint(1, 6),
        'subsample': uniform(0.6, 0.4),  # [0.6, 1.0]
        'colsample_bytree': uniform(0.6, 0.4),
        'learning_rate': uniform(0.01, 0.1),  # [0.01, 0.11]
        'n_estimators': [100, 200, 300]
    }

    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_dist,
        n_iter=20,
        scoring='balanced_accuracy',
        cv=3,
        verbose=2,
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(X_train, y_train, sample_weight=sample_weight)

    print(" Best Parameters:", random_search.best_params_)
    print(" Best CV Balanced Accuracy:", random_search.best_score_)


if __name__ == '__main__':
    train()