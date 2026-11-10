import numpy as np
from sklearn.datasets import load_wine, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_dataset(name, seed=42):
    if name == "small":
        # Wine: 178 samples, 13 features, 3 classes
        X, y = load_wine(return_X_y=True)
    elif name == "medium":
        # Breast cancer: 569 samples, 30 features, 2 classes
        X, y = load_breast_cancer(return_X_y=True)
    elif name == "large":
        # synthetic data: 100 features, only 15 of them are useful
        X, y = make_classification(n_samples=600, n_features=100, n_informative=15,
                                   n_redundant=10, n_classes=2, random_state=seed)
    else:
        raise ValueError("unknown dataset: " + name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test


DATASETS = {
    "small": "Wine (13 features)",
    "medium": "Breast Cancer (30 features)",
    "large": "Synthetic (100 features)",
}
