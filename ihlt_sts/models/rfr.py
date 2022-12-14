from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer


class RandomForestRegressorModel:


    def __init__(self):

        self.rfr = None


    def fit_search(self, X, y, n_estimators_list=None, max_depth_list=None):

        if n_estimators_list is None:
            n_estimators_list = [200, 300, 500]

        if max_depth_list is None:
            max_depth_list = [9, 10, 11, 12]

        param_grid_rfr = {
            'n_estimators': n_estimators_list,
            'max_depth': max_depth_list
        }

        pearson_scorer = make_scorer(lambda y_true, y_pred: pearsonr(y_true, y_pred)[0])

        rfr = RandomForestRegressor(
            criterion="squared_error"
        )

        CV_rfr = GridSearchCV(
            estimator=rfr,
            param_grid=param_grid_rfr,
            scoring=pearson_scorer,
            cv=5,
            verbose=2
        )

        self.rfr = CV_rfr.fit(X, y)


    def fit(self, X, y, n_estimators, max_depth):

        self.rfr = RandomForestRegressor(
            criterion="squared_error",
            n_estimators=n_estimators,
            max_depth=max_depth
        )

        self.rfr = self.rfr.fit(X, y)


    def predict(self, X):

        if self.rfr is None:
            raise ValueError("RandomForestRegressorModel.fit() must be called")

        y_pred = self.rfr.predict(X)

        return y_pred


    def get_best_params(self):

        if self.rfr is None:
            raise ValueError("RandomForestRegressorModel.fit() must be called")

        return self.rfr.best_params_


    def get_feature_importances(self):

        if self.rfr is None:
            raise ValueError("RandomForestRegressorModel.fit() must be called")

        return self.rfr.best_estimator_.feature_importances_
