from model_lib.Forest import default_random_forest
from model_lib.XGBoostR import default_XGBoost
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier


def ensemble(X_train, y_train):

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

    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', default_random_forest(X_train, y_train, train=False)),
            ('xgb', default_XGBoost(X_train, y_train, train=False)),
            ('mlp', mlp)
        ],
        voting='soft'
    )

    ensemble_model.fit(X_train, y_train)

    return ensemble_model
