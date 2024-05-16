from sklearn.svm import SVC

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import abstract_ml


class Svc(abstract_ml.AbstractMl):

    def __init__(self, is_last: bool):
        model_name = "Support Vector Classification"
        model_class = SVC
        test_grid = {
            'kernel': ['rbf'],            # Kernel type                  ['rbf', 'sigmoid']
            "C": [1000],                  # Regularization parameter     [0.1, 10, 1000, 10000]
        }
        super().__init__(is_last, model_name, test_grid, model_class)
