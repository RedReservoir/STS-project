import numpy as np

from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer

import warnings


class SupportVectorRegressorModel:
    """
    Wrapper class around the SVR (Support Vector Regressor) from sklearn.
    """


    def __init__(self):

        self.svr = None


    def fit_search(self, X, y, gamma_list=None, C_list=None, epsilon_list=None):
        """
        Searches the best parameters for the model.

        :param X: np.ndarray
            2D numpy array with the train data features to train with.
        :param y: np.ndarray
            1D numpy array with the train data target values.
        :param gamma_list: list, optional
            Values of parameter `gamma` to try.
        :param C_list: list, optional
            Values of parameter `C` to try.
        :param epsilon_list: list, optional
            Values of parameter `epsilon` to try.
        """

        if gamma_list is None:
            gamma_list = np.logspace(-6, -1, 6)

        if C_list is None:
            C_list = np.array([0.5, 1, 2, 4, 8, 10, 15, 20, 50, 100, 200, 375, 500, 1000])

        if epsilon_list is None:
            epsilon_list = np.linspace(0.1, 1, 10)

        svr_param = dict(
            gamma=gamma_list,
            C=C_list,
            epsilon=epsilon_list
        )

        pearson_scorer = make_scorer(lambda y_true, y_pred: pearsonr(y_true, y_pred)[0])

        svr = SVR(
            kernel='rbf'
        )

        CV_svr = GridSearchCV(
            estimator=svr,
            param_grid=svr_param,
            scoring=pearson_scorer,
            verbose=2
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.svr = CV_svr.fit(X, y)


    def fit(self, X, y, gamma, C, epsilon):
        """
        Searches the best parameters for the model.

        :param X: np.ndarray
            2D numpy array with the train data features to train with.
        :param y: np.ndarray
            1D numpy array with the train data target values.
        :param gamma: float
            Value of parameter `gamma` to pass to SVR.
        :param C: float
            Value of parameter `C` to pass to SVR.
        :param epsilon: float
            Value of parameter `epsilon` to pass to SVR.
        """

        self.svr = SVR(
            kernel='rbf',
            gamma=gamma,
            C=C,
            epsilon=epsilon
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.svr = self.svr.fit(X, y)


    def predict(self, X):
        """
        Predicts regression target values.

        :param X: np.ndarray
            2D numpy array with the train data features to predict with.

        :return: np.ndarray
            1D numpy array with the predicted values.
        """

        if self.svr is None:
            raise ValueError("SupportVectorRegressorModel.fit() must be called before SupportVectorRegressorModel.predict()")

        y_pred = self.svr.predict(X)

        return y_pred


    def get_best_params(self):
        """
        Returns the best parameters for the model.

        :return: dict
            Dictionary with the best SVR parameters.
        """

        if self.svr is None:
            raise ValueError("SupportVectorRegressorModel.fit() must be called before SupportVectorRegressorModel.predict()")

        return self.svr.best_params_