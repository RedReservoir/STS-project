class FullyConvolutionalNeuralNetworkModel:


    def __init__(self):

        self.fcnn = None


    def fit(self, X, y):
        raise NotImplementedError()


    def predict(self, X):
        raise NotImplementedError()


    def get_best_params(self, X):
        raise NotImplementedError()
