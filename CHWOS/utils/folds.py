import numpy as np
from sklearn.model_selection import StratifiedKFold  # type: ignore


def cross_validation_split(features_A, features_B, n_folds, seed=None):
    # Combine the features and create a target array
    features = np.concatenate((features_A, features_B), axis=0)
    targets = np.array([0] * len(features_A) + [1] * len(features_B))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_index, test_index in skf.split(features, targets):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = targets[train_index], targets[test_index]

        A_train = X_train[y_train == 0]
        B_train = X_train[y_train == 1]
        A_test = X_test[y_test == 0]
        B_test = X_test[y_test == 1]

        yield A_train, B_train, A_test, B_test
