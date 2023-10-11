from sklearn.tree import ExtraTreeClassifier
from ml_models.abstract_ml import AbstractMl



class ExtraTreeClass(AbstractMl):

    def __init__(self, X_train, X_test, y_train, y_test, is_last):
        self.model_name = "Extra Tree Classifier"
        self.eval_name = "Extra Tree Clas."
        self.model_class = ExtraTreeClassifier
        self.grid = {}
        self.grid_pmml = {}
        super().__init__(X_train, X_test, y_train, y_test, is_last)