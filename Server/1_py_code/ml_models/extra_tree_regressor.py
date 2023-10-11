from sklearn.tree import ExtraTreeRegressor
from ml_models.abstract_ml import AbstractMl



class ExtraTreeReg(AbstractMl):

    def __init__(self, X_train, X_test, y_train, y_test, is_last):
        self.model_name = "Extra Tree Regressor"
        self.eval_name = "Extra Tree Reg."
        self.model_class = ExtraTreeRegressor
        self.grid = {}
        self.grid_pmml = {}
        super().__init__(X_train, X_test, y_train, y_test, is_last)