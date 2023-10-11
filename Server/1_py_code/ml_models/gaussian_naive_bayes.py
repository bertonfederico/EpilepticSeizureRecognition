from sklearn.naive_bayes import GaussianNB
import numpy as np
from ml_models.abstract_ml import AbstractMl



class Gaussian(AbstractMl):

    def __init__(self, X_train, X_test, y_train, y_test, is_last):
        self.model_name = "Gaussian Naive Bayes"
        self.eval_name = "Gaussian NB"
        self.model_class = GaussianNB
        self.grid = {"var_smoothing": np.logspace(-9, 3, 13)}
        self.grid_pmml = {"var_smoothing": 10}
        super().__init__(X_train, X_test, y_train, y_test, is_last)