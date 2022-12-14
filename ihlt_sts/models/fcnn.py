from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class FullyConnectedNeuralNetworkModel:
    """
    Wrapper class around keras-built Sequential Neural Networks.
    """

    def __init__(self):

        self.fcnn = None
        self.history = None


    def fit(self, X, y, epochs=100, validation_split=0.1):
        """
        Creates the neural network and trains it.
        GD is used with ADAM optimizer. Loss function is MSE.

        :param X: np.ndarray
            2D numpy array with the train data features to train with.
        :param y: np.ndarray
            1D numpy array with the train data target values.
        :param epochs: int, optional
            Number of epochs to train the neural network for.
            Default is 100 epochs.
        :param validation_split: float, optional
            Percentage of the train data to use for validation.
            Default is 0.1 (10%).
        """

        self.fcnn = Sequential()
        self.fcnn.add(Dense(64, input_shape=(X.shape[1],), kernel_initializer='normal', activation='relu'))
        self.fcnn.add(Dense(32, kernel_initializer='normal', activation='relu'))
        self.fcnn.add(Dense(16, kernel_initializer='normal', activation='relu'))
        self.fcnn.add(Dense(8, kernel_initializer='normal', activation='relu'))
        self.fcnn.add(Dense(1, kernel_initializer='normal', activation='linear'))

        self.fcnn.compile(loss='mean_squared_error', optimizer='adam')
        self.history = self.fcnn.fit(X, y, epochs=epochs, validation_split=validation_split)


    def predict(self, X):
        """
        Predicts regression target values.

        :param X: np.ndarray
            2D numpy array with the train data features to predict with.

        :return: np.ndarray
            1D numpy array with the predicted values.
        """

        if self.fcnn is None:
            raise ValueError("FullyConnectedNeuralNetworkModel.fit() must be called")

        y_pred = self.fcnn.predict(X, verbose=0).flatten()

        return y_pred


    def get_history(self):
        """
        Getter for the neural network training history.

        :return: History
            The neural network training history.
        """

        if self.fcnn is None:
            raise ValueError("FullyConnectedNeuralNetworkModel.fit() must be called")

        return self.history


    def print_summary(self):
        """
        Prints the neural network summary.
        """

        if self.fcnn is None:
            raise ValueError("FullyConnectedNeuralNetworkModel.fit() must be called")

        self.fcnn.summary()
