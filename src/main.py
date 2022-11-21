"""
Main script
"""
from import_data import import_data


#
# Import data
#
df = import_data()

print(df.head())
print("Data set shape: ", df.shape)
