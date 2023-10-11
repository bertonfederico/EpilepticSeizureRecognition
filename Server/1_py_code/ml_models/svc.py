from sklearn.svm import SVC
import numpy as np
from ml_models.abstract_ml import AbstractMl




class Svc(AbstractMl):

    def __init__(self, X_train, X_test, y_train, y_test, is_last):
        self.model_name = "Support Vector Classification"
        self.eval_name = "SVC"
        self.model_class = SVC
        self.grid = {'kernel': ['rbf', 'poly'], "C" : np.logspace(-3, 3, 7)}
        self.grid_pmml = {'kernel': 'rbf', "C" : 10}
        super().__init__(X_train, X_test, y_train, y_test, is_last)