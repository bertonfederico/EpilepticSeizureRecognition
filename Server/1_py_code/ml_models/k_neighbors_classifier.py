from sklearn.neighbors import KNeighborsClassifier
from ml_models.abstract_ml import AbstractMl
import warnings
warnings.filterwarnings('ignore')



class KNeighborsClass(AbstractMl):

    def __init__(self, X_train, X_test, y_train, y_test, is_last):
        self.model_name = "k-nearest Neighbors"
        self.eval_name = "k-n Neighbors"
        self.model_class = KNeighborsClassifier
        self.grid = {
            "n_neighbors": [1, 100],
            "algorithm": ['auto', 'ball_tree', 'kd_tree', 'brute'],
            "leaf_size": [1, 100],
            "weights" : ['uniform', 'distance']
        }
        self.grid_pmml = {
            "n_neighbors": 100,
            "algorithm": 'auto',
            "leaf_size": 100,
            "weights" : 'uniform'
        }   
        super().__init__(X_train, X_test, y_train, y_test, is_last)