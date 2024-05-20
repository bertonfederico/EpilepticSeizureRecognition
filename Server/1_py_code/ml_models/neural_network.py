from sklearn.neural_network import MLPClassifier
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import abstract_ml


class NeuralNetwork(abstract_ml.AbstractMl):

    def __init__(self, is_last: bool):
        model_name = "Neural Network"
        model_class = MLPClassifier
        test_grid = {
            'hidden_layer_sizes': [(400, 400, 400, 400)],    # N° of layers and neurons     [(178, 178, 178), (400, 400, 400), (400, 400, 400, 400)]
            'activation': ['relu'],                          # Activation functions         ['logistic', 'relu', 'tanh']
            'solver': ['adam'],                              # Solvers                      ['sgd', 'adam']
            'alpha': [0.0001],                               # L2 Strength                  [0.0001, 0.01, 0.1]
            'max_iter': [100000]                             # Maximum iterations           [1000, 100000]
        }
        super().__init__(is_last, model_name, test_grid, model_class)
