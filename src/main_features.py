import setup_nltk

import pandas as pd

from data.semeval_sts import load_train_data

from features.features import get_all_features

if __name__ == "__main__":

    # Import semeval_sts_data

    train_df = load_train_data()

    print(train_df.head())
    print("Train dataset shape: ", train_df.shape)

    # Apply transform

    features_mat = get_all_features(train_df)
    features_df = pd.DataFrame(features_mat, index=None)

    print(features_df.head())
    print(features_df.describe())
