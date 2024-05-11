from sklearn.svm import SVC
from ml_models.abstract_ml import AbstractMl


class Svc(AbstractMl):

    def __init__(self, X_train, X_test, y_train, y_test, is_last):
        self.model_name = "Support Vector Classification"
        self.eval_name = "Support Vector Class."
        self.model_class = SVC
        self.grid = {
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],     # Kernel type
            "C": [0.01, 0.1, 1, 10],                            # Regularization parameter
        }
        self.grid_pmml = {
            'kernel': 'rbf',                                    # Kernel type
            "C": 10                                             # Regularization parameter
        }
        super().__init__(X_train, X_test, y_train, y_test, is_last)
