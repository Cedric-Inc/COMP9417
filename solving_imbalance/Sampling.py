import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks, EditedNearestNeighbours
from sklearn.preprocessing import StandardScaler


def data_sampler(X, y,
                 oversampling_methods=['SMOTE'],
                 undersampling_methods=['RANDOM'],
                 use_pca=False,
                 use_scaling=True,
                 n_components=100,
                 default_seed=0,
                 minority_samples=80,
                 majority_samples=500,
                 k_neighbors=3,
                 print_dis=False):

    '''
    oversampling_methods = ['SMOTE', 'BORDERLINE_SMOTE', 'ADASYN']
    undersampling_methods = ['RANDOM', 'NEARMISS', 'TOMEKLINKS', 'ENN']
    dimensionality reduction : PCA
    '''
    if print_dis:
        print("\n===  ===")
        class_counts_before = pd.Series(y).value_counts().sort_index()
        print(class_counts_before)
        print(f" {len(y)}")

    class_counts = pd.Series(y).value_counts()
    sampler_list = []

    if oversampling_methods:
        for method in oversampling_methods:
            method_upper = method.upper()
            if method_upper == 'SMOTE':
                print("Oversampling-SMOTE")
                sampler_list.append(
                    ('smote', SMOTE(
                        k_neighbors=k_neighbors, random_state=default_seed,
                        sampling_strategy={c: minority_samples for c,v in class_counts.items() if v <= minority_samples}
                    ))
                )
            elif method_upper == 'BORDERLINE_SMOTE':
                print("Oversampling-Borderline-SMOTE")
                # sampling_strategy_over = {c: minority_samples for c in range(28) if np.bincount(y, minlength=28)[c] < minority_samples}
                sampling_strategy_over = {c: minority_samples for c, v in class_counts.items() if v < minority_samples}
                sampler_list.append(
                    ('borderline_smote', BorderlineSMOTE(
                        sampling_strategy=sampling_strategy_over, random_state=default_seed, k_neighbors=k_neighbors
                    ))
                )
            elif method_upper == 'ADASYN':
                print("Oversampling-ADASYN")
                sampler_list.append(
                    ('adasyn', ADASYN(
                        sampling_strategy={c: minority_samples for c,v in class_counts.items() if v <= minority_samples},
                        n_neighbors=k_neighbors, random_state=default_seed
                    ))
                )
            else:
                print(f"Unknown method: {method}!!")
                sampler_list = []
                break

    if undersampling_methods:
        for method in undersampling_methods:
            method_upper = method.upper()
            if method_upper == 'RANDOM':
                print("Undersampling-Random UnderSampling")
                sampler_list.append(
                    ('under', RandomUnderSampler(
                        sampling_strategy={class_counts.idxmax(): majority_samples},
                        random_state=default_seed
                    ))
                )
            elif method_upper == 'NEARMISS':
                print("Undersampling-NearMiss")
                sampler_list.append(
                    ('nearmiss', NearMiss())
                )
            elif method_upper == 'TOMEKLINKS':
                print("Undersampling-Tomek Links ")
                sampler_list.append(
                    ('tomeklinks', TomekLinks())
                )
            elif method_upper == 'ENN':
                print("Undersampling-Edited Nearest Neighbours (ENN)")
                sampler_list.append(
                    ('enn', EditedNearestNeighbours())
                )
            else:
                print(f"Undersampling-Unknown Method: {method}!!")
                sampler_list = []
                break

    if sampler_list:
        pipeline = Pipeline(steps=sampler_list)
        X_resampled, y_resampled = pipeline.fit_resample(X, y)
    else:
        print("I stay.")
        X_resampled, y_resampled = X, y

    if use_scaling:
        print(f"StandardScaler")
        scaler = StandardScaler()
        X_resampled = scaler.fit_transform(X_resampled)

    if use_pca:
        pca = PCA(n_components=n_components)
        X_resampled = pca.fit_transform(X_resampled)

    if print_dis:
        print("\n===  ===")
        class_counts_after = pd.Series(y_resampled).value_counts().sort_index()
        print(class_counts_after)
        print(f" {len(y_resampled)}")

    return X_resampled, y_resampled
