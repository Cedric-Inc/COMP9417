from sklearn.tree import DecisionTreeClassifier

def default_decision_tree(X_train, y_train):
    model = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=20,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train, y_train)
    return model

