from typing import List, Tuple, Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import VotingClassifier, StackingClassifier


class WeightedEnsembleClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators: List[Tuple[str, BaseEstimator]], weights: Optional[List[float]] = None):
        self.estimators = estimators
        self.weights = weights
        self._vc: Optional[VotingClassifier] = None

    def fit(self, X, y):
        self._vc = VotingClassifier(estimators=self.estimators, voting='soft', weights=self.weights, n_jobs=None)
        self._vc.fit(X, y)
        return self

    def predict(self, X):
        if self._vc is None:
            raise ValueError("WeightedEnsembleClassifier is not fitted.")
        return self._vc.predict(X)

    def predict_proba(self, X):
        if self._vc is None:
            raise ValueError("WeightedEnsembleClassifier is not fitted.")
        if not hasattr(self._vc, 'predict_proba'):
            raise AttributeError("Underlying voting classifier does not support probability estimates.")
        return self._vc.predict_proba(X)

    def score(self, X, y):
        if self._vc is None:
            raise ValueError("WeightedEnsembleClassifier is not fitted.")
        return self._vc.score(X, y)


class StackingMetaLearner(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators: List[Tuple[str, BaseEstimator]], final_estimator: BaseEstimator, passthrough: bool = False, cv: int = 5):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.passthrough = passthrough
        self.cv = cv
        self._sc: Optional[StackingClassifier] = None

    def fit(self, X, y):
        self._sc = StackingClassifier(
            estimators=self.estimators,
            final_estimator=self.final_estimator,
            passthrough=self.passthrough,
            cv=self.cv,
            n_jobs=None
        )
        self._sc.fit(X, y)
        return self

    def predict(self, X):
        if self._sc is None:
            raise ValueError("StackingMetaLearner is not fitted.")
        return self._sc.predict(X)

    def predict_proba(self, X):
        if self._sc is None:
            raise ValueError("StackingMetaLearner is not fitted.")
        if not hasattr(self._sc, 'predict_proba'):
            raise AttributeError("Meta learner does not support probability estimates.")
        return self._sc.predict_proba(X)

    def score(self, X, y):
        if self._sc is None:
            raise ValueError("StackingMetaLearner is not fitted.")
        return self._sc.score(X, y)


