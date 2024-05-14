from sklearn.neural_network import MLPClassifier
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import abstract_ml


class NeuralNetwork(abstract_ml.AbstractMl):

    def __init__(self, is_last):
        model_name = "Neural Network"
        model_class = MLPClassifier
        test_grid = {
            'hidden_layer_sizes': [(178, 178, 178), (300, 300, 300)],  # N° of layers and neurons
            'activation': ['relu', 'tanh'],                            # Activation functions
            'max_iter': [1000, 100000]                                 # Maximum iterations
        }
        super().__init__(is_last, model_name, test_grid, model_class)
