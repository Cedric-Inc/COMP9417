#
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler  # 可选：特征标准化
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import pandas as pd
import numpy as np
from tqdm import tqdm  # 导入进度条库
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from solving_imbalance.Sampling import data_sampler
from model_lib.XGBoostR import default_XGBoost
from sklearn.preprocessing import OneHotEncoder


def compute_effective_number_weights(y, beta=0.9999):
    y = np.array(y)
    classes, counts = np.unique(y, return_counts=True)

    effective_num = (1.0 - np.power(beta, counts)) / (1.0 - beta)
    class_weights = 1.0 / effective_num
    class_weights = class_weights / np.sum(class_weights) * len(classes)

    class_weight_dict = dict(zip(classes, class_weights))
    return class_weight_dict


def weighted_log_los(y_true, y_pred):
    epsilon = 1e-12  #
    class_counts = np.sum(y_true, axis=0)
    class_weights = 1.0 / (class_counts + epsilon)
    class_weights /= np.sum(class_weights)

    sample_weights = np.sum(y_true * class_weights, axis=1)
    loss = -np.mean(sample_weights * np.sum(y_true * np.log(np.clip(y_pred, 1e-12, 1.0)), axis=1))
    return loss


def train():

    X = pd.read_csv('Data/X_train.csv')
    y = pd.read_csv('Data/y_train.csv')
    y = np.array(y).reshape(-1)  #

    #
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    X_train, y_train = data_sampler(X_train, y_train, oversampling_methods=['BORDERLINE_SMOTE', 'ADASYN'],
                                    undersampling_methods=['TOMEKLINKS'], minority_samples=100,
                                    majority_samples=400,use_pca=False, use_scaling=True)

    model = default_XGBoost(X_train, y_train, 'balanced')
        #

    encoder = OneHotEncoder(categories=[list(range(28))], sparse_output=False, handle_unknown='ignore')
    test_y2_onehot = encoder.fit_transform(y_test.reshape(-1, 1))

    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    print("\nTest Set Performance:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Test Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("entropy:", weighted_log_los(y_true=test_y2_onehot, y_pred=y_pred_proba))

    print('='*10, 'switch', '='*50)

    model = default_XGBoost(X_train, y_train, 'effective')

    # 
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    print("\nTest Set Performance:")
    print(classification_report(y_test, y_pred, digits=4))
    print("Test Balanced Accuracy:", balanced_accuracy_score(y_test, y_pred))
    print("entropy:", weighted_log_los(y_true=test_y2_onehot, y_pred=y_pred_proba))


if __name__ == '__main__':
    train()