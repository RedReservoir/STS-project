import setup_nltk

import pandas as pd

from data.semeval_sts import load_train_data

from features.features import get_all_features

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr


if __name__ == "__main__":

    # Load semeval_sts train data and calculate features

    train_df = load_train_data()

    y = train_df["gold-sim"]
    feature_mat = get_all_features(train_df)

    print("Features shape:", feature_mat.shape)
    print("Gold standard similarity values shape:", len(y))

    # Generate train/test split

    x_train, x_test, y_train, y_test = train_test_split(feature_mat, y, test_size=0.3, random_state=42)
    print("Train transform shape: ", x_train.shape)
    print("Train labels shape: ", y_train.shape)
    print("Test transform shape: ", x_test.shape)
    print("Test labels shape: ", y_test.shape)

    # Random Forest Regression

    rfr = RandomForestRegressor(random_state=42)

    param_grid = {
        'n_estimators': [200, 500],
        'criterion': ['squared_error', 'absolute_error'],
        'max_depth': [6, 7, 8]
    }

    CV_rfc = GridSearchCV(estimator=rfr, param_grid=param_grid, cv=5)
    CV_rfc.fit(x_train, y_train)

    print(CV_rfc.best_params_)
    pred = CV_rfc.predict(x_test)
    print("Accuracy for Random Forest on CV semeval_sts_data: ", pearsonr(y_test, pred))
