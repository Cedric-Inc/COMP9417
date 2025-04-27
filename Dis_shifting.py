from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, log_loss
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from model_lib.Forest import default_random_forest
from model_lib.XGBoostR import default_XGBoost
from model_lib.Network import default_neural_network
from solving_imbalance.Sampling import data_sampler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report


def weighted_log_los(y_true, y_pred):
    epsilon = 1e-12
    class_counts = np.sum(y_true, axis=0)
    class_weights = 1.0 / (class_counts + epsilon)
    class_weights /= np.sum(class_weights)

    sample_weights = np.sum(y_true * class_weights, axis=1)
    loss = -np.mean(sample_weights * np.sum(y_true * np.log(np.clip(y_pred, 1e-12, 1.0)), axis=1))
    return loss

def add_noise_df(df, noise_level=0.05, columns=None):

    df_noisy = df.copy()
    if columns is None:
        columns = df.columns

    noise = np.random.normal(loc=0.0, scale=noise_level, size=df[columns].shape)
    df_noisy[columns] = df[columns] + noise

    return df_noisy


def mask_features_df(df, mask_prob=0.3, mask_value='mean', columns=None):

    df_masked = df.copy()
    if columns is None:
        columns = df.columns

    for col in columns:
        mask = np.random.rand(len(df)) < mask_prob
        if mask_value == 'mean':
            fill_val = df[col].mean()
        elif mask_value == 'median':
            fill_val = df[col].median()
        else:
            fill_val = 0
        df_masked.loc[mask, col] = fill_val

    return df_masked


def run_pipeline(X_train, y_train, X_test, y_test, name=""):

    # 模型训练
    clf_rf = default_random_forest(X_train, y_train)
    clf_xgb = default_XGBoost(X_train, y_train)


    mlp = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation='relu',
        learning_rate_init=0.001,
        max_iter=200,
        batch_size=64,
        alpha=1e-4,
        early_stopping=True,
        random_state=42
    )

    # ensemble_model = StackingClassifier(
    #     estimators=[('rf', default_random_forest(X_train, y_train)),
    #                 ('xgb', default_XGBoost(X_train, y_train)),
    #                 ('mlp', mlp)],
    #     final_estimator=LogisticRegression()
    # )

    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', default_random_forest(X_train, y_train, train=False)),
            ('xgb', default_XGBoost(X_train, y_train, train=False)),
            ('mlp', mlp)
        ],
        voting='soft'
    )

    ensemble_model.fit(X_train, y_train)

    y_pred_rf = clf_rf.predict(X_test)
    y_pred_xgb = clf_xgb.predict(X_test)
    y_pred_voting = ensemble_model.predict(X_test)


    y_proba_rf = clf_rf.predict_proba(X_test)
    y_proba_xgb = clf_xgb.predict_proba(X_test)
    y_proba_voting = ensemble_model.predict_proba(X_test)


    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test_onehot = lb.transform(y_test)


    rf_loss = weighted_log_los(y_test_onehot, y_proba_rf)
    xgb_loss = weighted_log_los(y_test_onehot, y_proba_xgb)
    voting_loss = weighted_log_los(y_test_onehot, y_proba_voting)


    print(f"\n=== Results for {name} ===")

    print("\nRandom Forest:")
    print(classification_report(y_test, y_pred_rf, zero_division=0))
    print(f"Weighted Cross Entropy Loss: {rf_loss:.4f}")

    print("\nXGBoost:")
    print(classification_report(y_test, y_pred_xgb, zero_division=0))
    print(f"Weighted Cross Entropy Loss: {xgb_loss:.4f}")

    print("\nVoting Classifier:")
    print(classification_report(y_test, y_pred_voting, zero_division=0))
    print(f"Weighted Cross Entropy Loss: {voting_loss:.4f}")

    return {
        'rf_accuracy': classification_report(y_test, y_pred_rf, output_dict=True, zero_division=0)['accuracy'],
        'xgb_accuracy': classification_report(y_test, y_pred_xgb, output_dict=True, zero_division=0)['accuracy'],
        'voting_accuracy': classification_report(y_test, y_pred_voting, output_dict=True, zero_division=0)['accuracy'],
        'rf_loss': rf_loss,
        'xgb_loss': xgb_loss,
        'voting_loss': voting_loss
    }


def sample_split(X, y, test_size=0.2, random_state=42, method='original'):

    if method == 'noisy':
        X = add_noise_df(X, noise_level=0.08)
    elif method == 'occluded':
        X = mask_features_df(X, mask_prob=0.2, mask_value='mean')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train, y_train = data_sampler(
        X_train, y_train,
        oversampling_methods=None,
        undersampling_methods=None,
        use_pca=False,
        use_scaling=True
    )

    return X_train, y_train, X_test, y_test


if __name__ == '__main__':

    X = pd.read_csv('Data/X_train.csv')
    y = pd.read_csv('Data/y_train.csv').values.ravel()
    y = np.array(y).reshape(-1)

    results = {}
    # results['original'] = run_pipeline(X_train, y_train, X_test_orig, y_test_orig, name='Original Data')
    results['original'] = run_pipeline(*sample_split(X, y, method='original'), name='Original Data')
    # results['noisy'] = run_pipeline(X_train_noise, y_train, X_test_orig, y_test_orig, name='Noisy Data')
    results['noisy'] = run_pipeline(*sample_split(X, y, method='noisy'), name='Noisy Data')
    # results['occluded'] = run_pipeline(X_train_mask, y_train, X_test_orig, y_test_orig, name='Masked Data')
    results['occluded'] = run_pipeline(*sample_split(X, y, method='occluded'), name='Masked Data')

    pd.set_option('display.max_columns', None)

    result_df = pd.DataFrame(results)
    print("\n========== Summary ==========")
    print(result_df)