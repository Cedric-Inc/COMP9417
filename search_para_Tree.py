from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.model_selection import train_test_split
from solving_imbalance.Sampling import data_sampler


if __name__ == '__main__':

    X = pd.read_csv('Data/X_train.csv')
    y = pd.read_csv('Data/y_train.csv').values.ravel()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, y_train = data_sampler(X_train, y_train, oversampling_methods=['SMOTE', 'ADASYN'],
                                    undersampling_methods=['TOMEKLINKS'], minority_samples=100,
                                    majority_samples=400,use_pca=False, use_scaling=True)

    dtree = DecisionTreeClassifier(random_state=42)

    param_grid = {
        'max_depth': [10, 15, 20, 25],
        'criterion': ['gini', 'entropy'],
        'class_weight': ['balanced', None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    grid_search = GridSearchCV(
        estimator=dtree,
        param_grid=param_grid,
        scoring='accuracy',
        cv=5,
        n_jobs=-1,
        verbose=2
    )

    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)
