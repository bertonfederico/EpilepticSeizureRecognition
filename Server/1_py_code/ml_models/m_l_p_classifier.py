from sklearn.neural_network import MLPClassifier
from ml_models.abstract_ml import AbstractMl



class MLPClass(AbstractMl):

    def __init__(self, X_train, X_test, y_train, y_test, is_last):
        self.model_name = "Multi-layer Perceptron Classifier"
        self.eval_name = "MLPClassifier"
        self.model_class = MLPClassifier
        self.grid = {}
        self.grid_pmml = {}
        super().__init__(X_train, X_test, y_train, y_test, is_last)