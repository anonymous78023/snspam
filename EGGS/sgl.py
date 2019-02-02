"""
This class implements stacked graphical learning (SGL).
"""
import numpy as np
from . import utils
from . import print_utils
from sklearn.base import clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from collections import defaultdict


class SGL:
    """
    Stacked Graphical Learning (SGL).
    """

    def __init__(self, estimator, pr_func, relations, method='cv', folds=10, stacks=2):
        """
        Initialization of SGL classifier.

        Parameters
        ----------
        estimator : object
            Classifier object.
        pr_func : func (default=None)
            Domain-dependent helper method to generate pseudo-relational features.
        relations : list (default=None)
            Relations to use for relational modeling.
        folds : int (default=10)
            Number of folds to use for the cross-validation method.
        stacks : int (default=2)
            Number of stacks to use for SGL. Only relevant if sgl is not None.
        """
        self.estimator = estimator
        self.pr_func = pr_func
        self.relations = relations
        self.method = method
        self.folds = folds
        self.stacks = stacks

    def fit(self, X, y, target_col):
        """Performs stacked graphical learning using the cross-validation or holdout method."""

        X, y = check_X_y(X, y)

        if self.method == 'cv':
            self._cross_validation(X, y, target_col)
        else:
            self._holdout(X, y, target_col)

        return self

    def predict_proba(self, Xg, target_col):

        Xg = check_array(Xg)
        check_is_fitted(self, 'base_model_')
        check_is_fitted(self, 'stacked_models_')

        y_hat = self.base_model_.predict_proba(Xg)[:, 1]
        Xr, Xr_cols = self.pr_func(y_hat, target_col, self.relations)

        for stacked_model in self.stacked_models_:
            X = np.hstack([Xg, Xr])
            y_hat = stacked_model.predict_proba(X)[:, 1]
            Xr, Xr_cols = self.pr_func(y_hat, target_col, self.relations)

        y_score = np.hstack([1 - y_hat.reshape(-1, 1), y_hat.reshape(-1, 1)])
        return y_score

    def plot_feature_importance(self, X_cols):
        last_model = self.stacked_models_[-1]
        print_utils.print_model(last_model, X_cols)

    # private
    def _cross_validation(self, Xg, y, target_col):
        """Trains stacked learners on the entire training set using cross-validation."""

        Xr = None
        self.base_model_ = clone(self.estimator).fit(Xg, y)
        self.stacked_models_ = []

        for i in range(self.stacks):
            X = Xg if i == 0 else np.hstack([Xg, Xr])

            y_hat = utils.cross_val_predict(X, y, self.estimator, num_cvfolds=self.folds)
            Xr, Xr_cols = self.pr_func(y_hat, target_col, self.relations)

            X = np.hstack([Xg, Xr])
            clf = clone(self.estimator).fit(X, y)
            self.stacked_models_.append(clf)

        self.Xr_cols_ = Xr_cols

    def _holdout(self, Xg, y, target_col):
        """Sequentailly trains stacked learners with pseudo-relational features."""

        # data containers
        pr_features = defaultdict(dict)  # pr_features[i][j] is Xr for data piece j using predictions from model i

        # split data into equal-sized pieces
        Xg_array = np.array_split(Xg, self.stacks + 1)
        y_array = np.array_split(y, self.stacks + 1)
        target_col_array = np.array_split(target_col, self.stacks + 1)

        self.base_model_ = clone(self.estimator).fit(Xg, y)

        # fit a base model, and stacked models using pseudo-relational features
        for i in range(self.stacks + 1):
            X = Xg_array[i] if i == 0 else np.hstack([Xg_array[i], pr_features[i-1][i]])
            fit_model = clone(self.estimator).fit(X, y_array[i])
            self.stacked_models_.append(fit_model)

            # generate predictions for all subsequent data pieces
            for j in range(i + 1, self.stacks + 1):
                X = Xg_array[j] if j == 1 else np.hstack([Xg_array[j], pr_features[i-1][j]])
                y_hat = fit_model.predict_proba(X)[:, 1]
                Xr, Xr_cols = self.pr_func(y_hat, target_col_array[j], self.relations)
                pr_features[i][j] = Xr

        # delete base model trained on first data piece
        del self.stacked_models_[0]
        self.Xr_cols = Xr_cols
