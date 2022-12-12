import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer

import warnings


class SupportVectorRegressorModel:


    def __init__(self):

        self.svr = None


    def fit(self, X, y):

        gamma_list = np.logspace(-6, -1, 6)
        C_list = np.array([0.5, 1, 2, 4, 8, 10, 15, 20, 50, 100, 200, 375, 500, 1000])
        epsilon_list = np.linspace(0.1, 1, 10)

        svr_param = dict(
            gamma=gamma_list,
            C=C_list,
            epsilon=epsilon_list
        )

        pearson_scorer = make_scorer(lambda y_true, y_pred: pearsonr(y_true, y_pred)[0])

        CV_svr = GridSearchCV(
            estimator=SVR(kernel='rbf'),
            param_grid=svr_param,
            scoring=pearson_scorer
        )

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="correlation coefficient")

            self.svr = CV_svr.fit(X, y)


    def predict(self, X):

        if self.svr is None:
            raise ValueError("SupportVectorRegressorModel.fit() must be called before SupportVectorRegressorModel.predict()")

        y_pred = self.svr.predict(X)

        return y_pred


    def get_best_params(self):

        if self.svr is None:
            raise ValueError("SupportVectorRegressorModel.fit() must be called before SupportVectorRegressorModel.predict()")

        return self.svr.best_params_