from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer


class RandomForestRegressorModel:


    def __init__(self):

        self.rfr = None


    def fit(self, X, y):

        param_grid_rfr = {
            'n_estimators': [200, 300, 500],
            'max_depth': [6, 7, 8]
        }

        pearson_scorer = make_scorer(lambda y_true, y_pred: pearsonr(y_true, y_pred)[0])

        CV_rfr = GridSearchCV(
            estimator=RandomForestRegressor(criterion="squared_error"),
            param_grid=param_grid_rfr,
            scoring=pearson_scorer,
            cv=5
        )

        self.rfr = CV_rfr.fit(X, y)


    def predict(self, X):

        if self.rfr is None:
            raise ValueError("RandomForestRegressorModel.fit() must be called before RandomForestRegressorModel.predict()")

        y_pred = self.rfr.predict(X)

        return y_pred


    def get_best_params(self):

        if self.rfr is None:
            raise ValueError("RandomForestRegressorModel.fit() must be called before RandomForestRegressorModel.predict()")

        return self.rfr.best_params_