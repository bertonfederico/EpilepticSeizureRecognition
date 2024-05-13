from sklearn.neural_network import MLPClassifier
from ml_models.abstract_ml import AbstractMl


class NeuralNetwork(AbstractMl):

    def __init__(self, X_train, X_test, y_train, y_test, is_last):
        self.model_name = "Neural Network"
        self.eval_name = "Neural Network"
        self.model_class = MLPClassifier
        self.grid = {
            'hidden_layer_sizes': [(178, 178, 178), (300, 300, 300)],  # Testing with different n° of layers and neurons
            'activation': ['relu', 'tanh'],  # Testing different activation functions
            'max_iter': [1000, 100000]  # Maximum iterations
        }
        self.grid_pmml = {
            'hidden_layer_sizes': (300, 300, 300),  # Testing with different n° of layers and neurons
            'activation': 'relu',  # Testing different activation functions
            'max_iter': 100000  # Maximum iterations

        }
        super().__init__(X_train, X_test, y_train, y_test, is_last)
