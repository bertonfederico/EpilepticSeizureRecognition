from sklearn.tree import DecisionTreeRegressor
from ml_models.abstract_ml import AbstractMl



class DecisionTreeReg(AbstractMl):

    def __init__(self, X_train, X_test, y_train, y_test, is_last):
        self.model_name = "Decision Tree Regressor"
        self.eval_name = "Decision Tree Reg."
        self.model_class = DecisionTreeRegressor
        self.grid = {}
        self.grid_pmml = {}
        super().__init__(X_train, X_test, y_train, y_test, is_last)