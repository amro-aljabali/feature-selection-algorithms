import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold


class FeatureSelectionProblem:
    """
    Solution = binary vector of length n_features (1 = feature is used).
    Fitness (minimize) = alpha * error_rate + (1 - alpha) * (selected / total)
    The error rate comes from 3-fold cross validation of a KNN (k=5)
    on the training set only. The test set is kept for the final check.
    """

    def __init__(self, X_train, y_train, alpha=0.99, k=5, folds=3):
        self.X = X_train
        self.y = y_train
        self.n_features = X_train.shape[1]
        self.alpha = alpha
        self.k = k
        self.cache = {}
        self.evaluations = 0

        n_classes = y_train.max() + 1
        skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=0)

        # KNN is the slow part, so we prepare each fold once.
        # diff[v, t, f] = (x_val[v, f] - x_train[t, f])^2
        # then the distance for any subset is just diff @ mask
        self.folds = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            Xt, Xv = X_train[train_idx], X_train[val_idx]
            diff = ((Xv[:, None, :] - Xt[None, :, :]) ** 2).astype(np.float32)
            diff = diff.reshape(-1, self.n_features)
            onehot = np.eye(n_classes)[y_train[train_idx]]
            self.folds.append((diff, len(val_idx), onehot, y_train[val_idx]))

    def error_rate(self, mask):
        m = mask.astype(np.float32)
        errors = []
        for diff, n_val, onehot, y_val in self.folds:
            dist = (diff @ m).reshape(n_val, -1)
            nearest = np.argpartition(dist, self.k, axis=1)[:, :self.k]
            votes = onehot[nearest].sum(axis=1)
            pred = votes.argmax(axis=1)
            errors.append(np.mean(pred != y_val))
        return float(np.mean(errors))

    def fitness(self, mask):
        mask = np.asarray(mask, dtype=int)
        if mask.sum() == 0:
            return 1.0  # empty subset is the worst possible solution

        key = mask.tobytes()
        if key in self.cache:
            return self.cache[key]

        self.evaluations += 1
        err = self.error_rate(mask)
        ratio = mask.sum() / self.n_features
        f = self.alpha * err + (1 - self.alpha) * ratio
        self.cache[key] = f
        return f


def test_accuracy(mask, X_train, y_train, X_test, y_test, k=5):
    cols = np.flatnonzero(mask)
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train[:, cols], y_train)
    return float(np.mean(knn.predict(X_test[:, cols]) == y_test))
