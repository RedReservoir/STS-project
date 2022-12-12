from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import pearsonr
from sklearn.metrics import make_scorer


class RandomForestModel:


    def __init__(self):

        self.rfr = None


    def fit(self, X, y):

        # Instantiate model and find best parameters

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

        CV_rfr.fit(X, y)

        self.rfr = RandomForestRegressor(criterion="squared_error", **CV_rfr.get_params())

        self.rfr.fit(X, y)


    def predict(self, X):

        if self.rfr is None:
            raise ValueError("RandomForestModel.fit() must be called before RandomForestModel.predict()")

        y_pred = self.rfr.predict(X)

        return y_pred

