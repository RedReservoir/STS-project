from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


class FullyConvolutionalNeuralNetworkModel:


    def __init__(self):

        self.fcnn = None
        self.history = None


    def fit(self, X, y):

        self.fcnn = Sequential()
        self.fcnn.add(Dense(64, input_shape=(30,), kernel_initializer='normal', activation='relu'))
        self.fcnn.add(Dense(32, kernel_initializer='normal', activation='relu'))
        self.fcnn.add(Dense(16, kernel_initializer='normal', activation='relu'))
        self.fcnn.add(Dense(8, kernel_initializer='normal', activation='relu'))
        self.fcnn.add(Dense(1, kernel_initializer='normal', activation='linear'))

        self.fcnn.compile(loss='mean_squared_error', optimizer='adam')
        self.history = self.fcnn.fit(X, y, epochs=100, validation_split=0.1)


    def predict(self, X):

        if self.fcnn is None:
            raise ValueError("FullyConvolutionalNeuralNetworkModel.fit() must be called")

        y_pred = self.fcnn.predict(X)

        return y_pred


    def get_history(self):

        if self.fcnn is None:
            raise ValueError("FullyConvolutionalNeuralNetworkModel.fit() must be called")

        return self.history


    def print_summary(self):

        if self.fcnn is None:
            raise ValueError("FullyConvolutionalNeuralNetworkModel.fit() must be called")

        self.fcnn.summary()
