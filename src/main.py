"""
Main script
"""
import pandas as pd

from import_data import import_data
from similarities import *
from features import *


#
# Import data
#
df = import_data()

print(df.head())
print("Data set shape: ", df.shape)


#
# Apply features
#
features_mat = get_features(df)
features_df = pd.DataFrame(features_mat)

print(features_df.head())
print(features_df.describe())
