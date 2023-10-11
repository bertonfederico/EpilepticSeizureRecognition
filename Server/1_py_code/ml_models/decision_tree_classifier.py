from sklearn.tree import DecisionTreeClassifier
from ml_models.abstract_ml import AbstractMl



class DecisionTreeClass(AbstractMl):

    def __init__(self, X_train, X_test, y_train, y_test, is_last):
        self.model_name = "Decision Tree Classifier"
        self.eval_name = "Decision Tree Clas."
        self.model_class = DecisionTreeClassifier
        self.grid = {}
        self.grid_pmml = {}
        super().__init__(X_train, X_test, y_train, y_test, is_last)