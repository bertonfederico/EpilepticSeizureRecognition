from sklearn.neural_network import MLPClassifier
from ml_models.abstract_ml import AbstractMl




class NeuralNetwork(AbstractMl):

    def __init__(self, X_train, X_test, y_train, y_test, is_last):
        self.model_name = "Neural Network"
        self.eval_name = "Neural Network"
        self.model_class = MLPClassifier
        self.grid = {
            'hidden_layer_sizes': [(20,), (100, 100)],
            'activation': ['tanh', 'relu'],
            'solver': ['sgd'],
            'alpha': [0.0001, 0.05],
            'max_iter': [1000]
        }
        self.grid_pmml = {
            'hidden_layer_sizes': (100, 100),
            'activation': 'tanh',
            'solver': 'sgd',
            'alpha': 0.0001,
            'max_iter': 1000
        }
        super().__init__(X_train, X_test, y_train, y_test, is_last)