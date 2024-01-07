from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer as mice

from common.imputeMethods import ImputeMethod
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class DecisionTreeImputerClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, random_state=42, impute_method=ImputeMethod.SEMI_GLOBAL):
        self.tree = DecisionTreeClassifier(random_state=random_state)
        self.impute_method = impute_method
        self.imputer = mice()

    def fit(self, X, y, sample_weight=None, check_input=True) -> None:
        if self.impute_method == ImputeMethod.GLOBAL:
            X = self.imputer.fit_transform(X)
        self.tree.fit(X, y, sample_weight=sample_weight, check_input=check_input)

    def predict(self, X, check_input=True):
        if self.impute_method == ImputeMethod.GLOBAL:
            X = self.imputer.transform(X)
        return self.tree.predict(X, check_input=check_input)

    def predict_proba(self, X, check_input=True):
        if self.impute_method == ImputeMethod.GLOBAL:
            X = self.imputer.transform(X)
        return self.tree.predict_proba(X, check_input=check_input)
