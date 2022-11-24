"""
Train a random forest regression model to determine the final similarity value
"""
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr

from import_data import import_data
from similarities import *
from features import *


#
# Import data and generate features
#
data = import_data()
y = data["gold-sim"]
feature_mat = get_features(data)

print("Features shape: ", feature_mat)
print("Gold standard similarity values shape: ", y)

# Train - test split
x_train, x_test, y_train, y_test = train_test_split(feature_mat, y, test_size=0.3, random_state=42)
print("Train features shape: ", x_train.shape)
print("Train labels shape: ", y_train.shape)
print("Test features shape: ", x_test.shape)
print("Test labels shape: ", y_test.shape)


#
# Random Forest Regression
#
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
print("Accuracy for Random Forest on CV data: ", pearsonr(y_test, pred))
