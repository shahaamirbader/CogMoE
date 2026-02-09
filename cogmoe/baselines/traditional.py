"""
Traditional ML baselines for cognitive load classification.

Provides sklearn/xgboost wrappers for comparison against CogMoE in
Paper Tables 3-7. Input: pre-extracted feature vectors (89 dimensions =
33 ECG + 16 EEG + 30 Gaze + 10 EDA). Output: binary CL label (0=low, 1=high).

Usage:
    model = RandomForestBaseline()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


class RandomForestBaseline:
    """
    Random Forest classifier baseline.

    Paper Tables 3-7: RF with 100 estimators, default sklearn hyperparameters.
    Operates on the concatenated 89-dim feature vector.

    Args:
        **kwargs: Forwarded to sklearn.ensemble.RandomForestClassifier.
            Defaults: n_estimators=100, random_state=42.
    """

    def __init__(self, **kwargs):
        defaults = {
            "n_estimators": 100,
            "random_state": 42,
            "n_jobs": -1,
        }
        defaults.update(kwargs)
        self.model = RandomForestClassifier(**defaults)

    def fit(self, X, y):
        """
        Fit the Random Forest on training data.

        Args:
            X: array-like of shape (n_samples, 89) - concatenated features.
            y: array-like of shape (n_samples,) - binary labels {0, 1}.

        Returns:
            self
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict binary class labels.

        Args:
            X: array-like of shape (n_samples, 89).

        Returns:
            np.ndarray of shape (n_samples,) - predicted labels {0, 1}.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: array-like of shape (n_samples, 89).

        Returns:
            np.ndarray of shape (n_samples, 2) - probabilities for [class 0, class 1].
        """
        return self.model.predict_proba(X)


class XGBoostBaseline:
    """
    XGBoost gradient-boosted tree classifier baseline.

    Paper Tables 3-7: XGBClassifier with default hyperparameters.
    Requires the xgboost package to be installed.

    Args:
        **kwargs: Forwarded to xgboost.XGBClassifier.
            Defaults: n_estimators=100, max_depth=6, learning_rate=0.1,
            use_label_encoder=False, eval_metric='logloss', random_state=42.

    Raises:
        ImportError: If xgboost is not installed.
    """

    def __init__(self, **kwargs):
        if XGBClassifier is None:
            raise ImportError(
                "xgboost is required for XGBoostBaseline. "
                "Install with: pip install xgboost"
            )
        defaults = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "use_label_encoder": False,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": -1,
        }
        defaults.update(kwargs)
        self.model = XGBClassifier(**defaults)

    def fit(self, X, y):
        """
        Fit the XGBoost model on training data.

        Args:
            X: array-like of shape (n_samples, 89) - concatenated features.
            y: array-like of shape (n_samples,) - binary labels {0, 1}.

        Returns:
            self
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict binary class labels.

        Args:
            X: array-like of shape (n_samples, 89).

        Returns:
            np.ndarray of shape (n_samples,) - predicted labels {0, 1}.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: array-like of shape (n_samples, 89).

        Returns:
            np.ndarray of shape (n_samples, 2) - probabilities for [class 0, class 1].
        """
        return self.model.predict_proba(X)


class MLPBaseline:
    """
    Multi-Layer Perceptron classifier baseline using sklearn.

    Paper Tables 3-7: MLPClassifier with hidden layers (256, 128), Adam optimizer.
    Operates on the concatenated 89-dim feature vector.

    Args:
        **kwargs: Forwarded to sklearn.neural_network.MLPClassifier.
            Defaults: hidden_layer_sizes=(256, 128), activation='relu',
            solver='adam', max_iter=500, random_state=42.
    """

    def __init__(self, **kwargs):
        defaults = {
            "hidden_layer_sizes": (256, 128),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 500,
            "random_state": 42,
            "early_stopping": True,
            "validation_fraction": 0.1,
        }
        defaults.update(kwargs)
        self.model = MLPClassifier(**defaults)

    def fit(self, X, y):
        """
        Fit the MLP on training data.

        Args:
            X: array-like of shape (n_samples, 89) - concatenated features.
            y: array-like of shape (n_samples,) - binary labels {0, 1}.

        Returns:
            self
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict binary class labels.

        Args:
            X: array-like of shape (n_samples, 89).

        Returns:
            np.ndarray of shape (n_samples,) - predicted labels {0, 1}.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: array-like of shape (n_samples, 89).

        Returns:
            np.ndarray of shape (n_samples, 2) - probabilities for [class 0, class 1].
        """
        return self.model.predict_proba(X)


class KNNBaseline:
    """
    K-Nearest Neighbors classifier baseline.

    Paper Tables 3-7: KNeighborsClassifier with k=5.
    Operates on the concatenated 89-dim feature vector.

    Args:
        **kwargs: Forwarded to sklearn.neighbors.KNeighborsClassifier.
            Defaults: n_neighbors=5, n_jobs=-1.
    """

    def __init__(self, **kwargs):
        defaults = {
            "n_neighbors": 5,
            "n_jobs": -1,
        }
        defaults.update(kwargs)
        self.model = KNeighborsClassifier(**defaults)

    def fit(self, X, y):
        """
        Fit the KNN model on training data.

        Args:
            X: array-like of shape (n_samples, 89) - concatenated features.
            y: array-like of shape (n_samples,) - binary labels {0, 1}.

        Returns:
            self
        """
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """
        Predict binary class labels.

        Args:
            X: array-like of shape (n_samples, 89).

        Returns:
            np.ndarray of shape (n_samples,) - predicted labels {0, 1}.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: array-like of shape (n_samples, 89).

        Returns:
            np.ndarray of shape (n_samples, 2) - probabilities for [class 0, class 1].
        """
        return self.model.predict_proba(X)
