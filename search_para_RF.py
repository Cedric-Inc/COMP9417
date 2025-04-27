import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from itertools import product
from solving_imbalance.Sampling import data_sampler


param_grid = {
    'n_estimators': [300, 500, 700],
    'max_depth': [8, 10, 12],
    'min_samples_leaf': [5, 10, 20],
    'max_features': [0.3, 'sqrt']
}

def train_and_evaluate_rf(X_train, y_train, X_test, y_test, params):
    model = RandomForestClassifier(
        n_estimators=params['n_estimators'],
        max_depth=params['max_depth'],
        min_samples_leaf=params['min_samples_leaf'],
        max_features=params['max_features'],
        class_weight='balanced_subsample',
        bootstrap=True,
        oob_score=True,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = balanced_accuracy_score(y_test, preds)
    return acc, model.oob_score_

if __name__ == '__main__':

    X = pd.read_csv('Data/X_train.csv')
    y = pd.read_csv('Data/y_train.csv').values.ravel()


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, y_train = data_sampler(X_train, y_train, oversampling_methods=['SMOTE', 'ADASYN'],
                                    undersampling_methods=['TOMEKLINKS'], minority_samples=100,
                                    majority_samples=400,use_pca=False, use_scaling=True)

    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file_path = f"rf_gridsearch_log_{now}.txt"

    with open(log_file_path, "w") as f:
        f.write("==== Random Forest Hyperparameter Search Log ====\n\n")


    param_combinations = list(product(
        param_grid['n_estimators'],
        param_grid['max_depth'],
        param_grid['min_samples_leaf'],
        param_grid['max_features']
    ))

    best_score = 0
    best_params = None

    print(f"Total combinations to try: {len(param_combinations)}\n")

    for idx, (n_estimators, max_depth, min_samples_leaf, max_features) in enumerate(param_combinations):
        params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features
        }
        print(f"Trying combination {idx+1}/{len(param_combinations)}: {params}")

        acc, oob = train_and_evaluate_rf(X_train, y_train, X_test, y_test, params)

        log_text = (
            f"Combination {idx+1}: {params}\n"
            f"Balanced Accuracy on Test Set: {acc:.4f}\n"
            f"OOB Score: {oob:.4f}\n\n"
        )
        print(log_text)

        with open(log_file_path, "a") as f:
            f.write(log_text)

        if acc > best_score:
            best_score = acc
            best_params = params


    summary = (
        "\n\n=== Best Hyperparameters Found ===\n"
        f"Best Params: {best_params}\n"
        f"Best Balanced Accuracy: {best_score:.4f}\n"
    )
    print(summary)

    with open(log_file_path, "a") as f:
        f.write(summary)
