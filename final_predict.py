from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import torch
from model_lib.Network import default_neural_network
from solving_imbalance.Sampling import data_sampler
from model_lib.Ensemble import ensemble

def weighted_log_los(y_true, y_pred):
    epsilon = 1e-12
    class_counts = np.sum(y_true, axis=0)
    class_weights = 1.0 / (class_counts + epsilon)
    class_weights /= np.sum(class_weights)

    sample_weights = np.sum(y_true * class_weights, axis=1)
    loss = -np.mean(sample_weights * np.sum(y_true * np.log(np.clip(y_pred, 1e-12, 1.0)), axis=1))
    return loss


if __name__ == '__main__':

    X = pd.read_csv('Data/X_train.csv')
    y = pd.read_csv('Data/y_train.csv')
    y = np.array(y).reshape(-1)

    X, y = data_sampler(X, y, oversampling_methods=['SMOTE', 'ADASYN'], undersampling_methods=['TOMEKLINKS'],
                     minority_samples=100, majority_samples=400,use_pca=False, use_scaling=True)

    X = torch.FloatTensor(X)
    model = default_neural_network(X, y, n_epochs=110)

    testset_x1 = pd.read_csv('Data/X_test_1.csv')
    testset_x1 = torch.FloatTensor(testset_x1.values).to('cuda')
    y_proba1 = model.predict_proba(testset_x1).cpu().numpy()

    np.save('BETA/preds_1.npy', y_proba1)

    # Use RF in Distribution shifting

    model = ensemble(X, y)

    testset_x2 = pd.read_csv('Data/X_test_2.csv')
    y_proba2 = model.predict_proba(testset_x2[202:])  # (1818, 28)

    np.save('BETA/preds_2.npy', y_proba1)

    y_test_1_ohe = (np.arange(28) == np.random.choice(28, size=1000)[:, None]).astype(int)
    y_test_2_ohe = (np.arange(28) == np.random.choice(28, size=1818)[:, None]).astype(int)

    score1 = weighted_log_los(y_test_1_ohe, y_proba1)
    score2 = weighted_log_los(y_test_2_ohe, y_proba2)

    print('score1:', score1, '; score2:', score2)