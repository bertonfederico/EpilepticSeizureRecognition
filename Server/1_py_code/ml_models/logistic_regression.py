from sklearn.linear_model import LogisticRegression
from ml_models.abstract_ml import AbstractMl


class LogisticReg(AbstractMl):

    def __init__(self, X_train, X_test, y_train, y_test, is_last):
        self.model_name = "Logistic regression"
        self.eval_name = "LogReg"
        self.model_class = LogisticRegression
        self.grid = {
            "penalty": [None, 'l1', 'l2'],                      # Penalty type (none, lasso, ridge)
            'max_iter': [1000, 10000]                           # Maximum iterations
        }
        self.grid_pmml = {
            "penalty": [None],                                  # Penalty type (none, lasso, ridge)
            'max_iter': [1000]                                  # Maximum iterations
        }
        super().__init__(X_train, X_test, y_train, y_test, is_last)
