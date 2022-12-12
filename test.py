from ihlt_sts.models.model_test import RandomForestModel
from ihlt_sts.data.semeval import load_train_data, load_test_data
from features.features import get_all_features

from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":

    #

    print("LOG: Load data")

    train_data_df = load_train_data()
    test_data_df = load_test_data()

    y_train = train_data_df["gold-sim"]
    y_test = test_data_df["gold-sim"]

    print("LOG: Get features")

    train_features_df = get_all_features(train_data_df)
    test_features_df = get_all_features(test_data_df)

    print("LOG: Scale features")

    scaler = StandardScaler()
    scaler.fit(train_features_df)

    train_scaled_features_df = scaler.transform(train_features_df)
    test_scaled_features_df = scaler.transform(test_features_df)

    #

    rfm = RandomForestModel()

    print("LOG: Fit")

    rfm.fit(train_scaled_features_df, y_train)

    print("LOG: Predict")

    y_pred = rfm.predict(test_scaled_features_df)