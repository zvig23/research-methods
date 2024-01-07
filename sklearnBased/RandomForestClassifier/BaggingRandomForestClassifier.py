import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer as mice

from common.imputeMethods import ImputeMethod
from sklearn.exceptions import ConvergenceWarning
import warnings
from sklearn.ensemble import VotingClassifier

from sklearnBased.DecisionTreeClassifier.SimpleDecisionTreeClassifier import DecisionTreeImputerClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)


class BaggingRandomForestClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, n_estimators=100, criterion='gini', random_state=42,
                 impute_method=ImputeMethod.GLOBAL):
        self.forest = None
        self.impute_method = impute_method
        self.imputer = mice()
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.random_state = random_state

    def bootstrap_sample(self, X, Y):
        """
        Function that creates a bootstraped sample with the class instance parameters

        Parameters:
        - X: Features
        - Y: Labels

        Returns:
        - X_bootstrap: Bootstrapped features
        - Y_bootstrap: Bootstrapped labels
        """
        # Set seed for reproducibility
        self.rng = np.random.default_rng(seed=0)

        # Get the number of samples in the original dataset
        n_samples = X.shape[0]

        # Create an array to store indices for the bootstrap samples
        bootstrap_indices = self.rng.choice(n_samples, size=(100, n_samples), replace=True)

        # Initialize lists to store bootstrapped samples
        X_bootstrap = []
        Y_bootstrap = []

        # Generate bootstrap samples
        for indices in bootstrap_indices:
            X_bootstrap.append(X[indices])
            Y_bootstrap.append(Y[indices])

        return np.array(X_bootstrap), np.array(Y_bootstrap)

    def fit(self, X, y) -> None:
        if self.impute_method == ImputeMethod.GLOBAL:
            X = self.imputer.fit_transform(X)
        estimators = []
        for tree_index in range(self.n_estimators):
            x_bootstrap, y_bootstrap = self.bootstrap_sample(X, y)
            if self.impute_method == ImputeMethod.SEMI_GLOBAL:
                fitted_tree = DecisionTreeImputerClassifier(impute_method=ImputeMethod.GLOBAL)
            else:
                fitted_tree = DecisionTreeImputerClassifier(impute_method=ImputeMethod.LOCAL)
            tree_name = f'dt{tree_index}'
            fitted_tree.fit(x_bootstrap, y_bootstrap)
            estimators.append((tree_name, fitted_tree))
        self.forest = VotingClassifier(estimators=estimators, voting="soft")
        self.forest.fit(X, y, sample_weight=0)

    def predict(self, X, check_input=True):
        if self.impute_method == ImputeMethod.GLOBAL:
            X = self.imputer.transform(X)
        return self.forest.predict(X, check_input=check_input)

    def predict_proba(self, X):
        if self.impute_method == ImputeMethod.GLOBAL:
            X = self.imputer.transform(X)
        return self.forest.predict_proba(X)
