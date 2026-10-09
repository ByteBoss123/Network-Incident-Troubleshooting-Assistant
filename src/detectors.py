"""Object-oriented detector interface.

Every detector implements the same contract (fit on train, score, threshold from a train percentile, flag),
so the evaluation loop, the Airflow task and the API can swap detectors without special cases:

    Detector (abstract)        fit(X) -> self, score(X) -> array, flag(X) -> bool array
    ├── RuleDetector           exception-template rule (no training)
    ├── PCAResidualDetector    standardize -> PCA (95% variance) -> squared reconstruction error
    └── IsolationForestDetector

`evaluate()` runs any list of detectors on the chronological split; tests check it reproduces
results/detection_metrics.json exactly, so the refactor did not change a single prediction.
"""
from abc import ABC, abstractmethod

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from detect import EXCEPTION_EVENTS, SEED, chrono_split, load_matrix, metrics


class Detector(ABC):
    name = "detector"
    percentile = 96.0  # fixed in advance from the ~4% anomaly base rate, never tuned on test labels

    def __init__(self):
        self.threshold = None

    def fit(self, X):
        self._fit(X)
        self.threshold = float(np.percentile(self.score(X), self.percentile))
        return self

    def flag(self, X):
        if self.threshold is None:
            raise RuntimeError(f"{self.name}: call fit() first")
        return self.score(X) > self.threshold

    def _fit(self, X):
        pass

    @abstractmethod
    def score(self, X):
        ...


class RuleDetector(Detector):
    name = "rule_exception_templates"

    def __init__(self, columns):
        super().__init__()
        self.idx = [i for i, c in enumerate(columns) if c in EXCEPTION_EVENTS]

    def fit(self, X):
        self.threshold = 0.0  # any exception template at all
        return self

    def score(self, X):
        return X[:, self.idx].sum(1)


class PCAResidualDetector(Detector):
    name = "pca_residual"

    def _fit(self, X):
        self.scaler = StandardScaler().fit(X)
        self.pca = PCA(n_components=0.95, random_state=SEED).fit(self.scaler.transform(X))

    def score(self, X):
        Z = self.scaler.transform(X)
        return ((Z - self.pca.inverse_transform(self.pca.transform(Z))) ** 2).sum(1)


class IsolationForestDetector(Detector):
    name = "isolation_forest"

    def _fit(self, X):
        self.model = IsolationForest(n_estimators=300, random_state=SEED).fit(X)

    def score(self, X):
        return -self.model.score_samples(X)


def evaluate(make_detectors=None):
    X, y, _ = load_matrix()
    tr, te = chrono_split(len(y))
    Xtr, Xte, yte = X.values[tr], X.values[te], y[te]
    dets = (make_detectors or default_detectors)(list(X.columns))
    out = {}
    for d in dets:
        d.fit(Xtr)
        out[d.name] = metrics(yte, d.score(Xte), d.flag(Xte))
    return out


def default_detectors(columns):
    return [RuleDetector(columns), PCAResidualDetector(), IsolationForestDetector()]
