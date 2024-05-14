from typing import Type, Dict, Any

import dataframe_image as dfi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline


class AbstractMl(object):

    def __init__(self, is_last: bool, model_name: str, test_grid: dict, model_class: Type):
        print('\n\n################################# ' + model_name + ' #################################')
        self.is_last_ml_algorithm = is_last
        self.restr_name = model_name.replace(" ", "")
        self.model_class = model_class
        self.test_grid = test_grid
        self.eval = ['Accuracy:', 'Balanced Accuracy:', 'Precision:', 'Recall:', 'F1:', 'ROC curve:']
        dictionary = {'Evaluation': self.eval}
        self.total_df = pd.DataFrame(dictionary)

    def train_assessment_phase(self, X_train: np.ndarray, y_train: np.ndarray,
                               X_development: np.ndarray, y_development: np.ndarray):
        """
        Train & test phase for hyperparameters

        :param X_train: features dataset for model training
        :param y_train: y dataset for model training
        :param X_development: features dataset for test prediction
        :param y_development: y dataset for prediction checking
        :return: best_params_
        """

        Y_hat_test, best_params_ = self.train_assessment(X_train, y_train, X_development)
        self.confusion_matrix(y_development, Y_hat_test, False)
        self.evaluation(y_development, Y_hat_test, False)
        return best_params_

    def train_assessment(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray):
        """ preparing """
        cross_val = GridSearchCV(self.model_class(), self.test_grid, cv=7, verbose=10)

        """ training """
        cross_val.fit(X_train, y_train)

        """ test """
        Y_hat_test = cross_val.predict(X_test)

        """ printing hyperparameters results """
        print('Best hyperparameters: ', cross_val.best_params_)

        return Y_hat_test, cross_val.best_params_

    def final_train_test_phase(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                               y_test: np.ndarray, best_params_set: Dict[str, Any]):
        """
        Final train & test phase

        :param X_train: features dataset for model training
        :param y_train: y dataset for model training
        :param X_test: features dataset for test prediction
        :param y_test: y dataset for prediction checking
        :param best_params_set: hyperparameters set by @train_assessment
        """

        Y_hat_test = self.final_train_test(X_train, y_train, X_test, best_params_set)
        self.confusion_matrix(y_test, Y_hat_test, True)
        self.evaluation(y_test, Y_hat_test, True)

    def final_train_test(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                         best_params_set: Dict[str, Any]):
        """ preparing """
        new_classifier = self.model_class(**best_params_set)

        """ training """
        new_classifier.fit(X_train, y_train)

        """ test """
        Y_hat_test = new_classifier.predict(X_test)

        return Y_hat_test

    def confusion_matrix(self, y_test: np.ndarray, Y_hat_test: np.ndarray, is_final_test: bool):
        """
        Confusion matrix

        :param y_test: y value of test part dataset
        :param Y_hat_test: predicted y value for test part dataset
        :param is_final_test: True if it's the final test, false if it's a development test
        """

        print('####### Confusion matrix ######')
        cm = confusion_matrix(y_test, Y_hat_test)

        """ results """
        print("Non-epileptic classified as non-epileptic: ", cm[0, 0])
        print("Non-epileptic classified as epileptic: ", cm[0, 1])
        print("Epileptic classified as non-epileptic: ", cm[1, 0])
        print("Epileptic classified as epileptic: ", cm[1, 1])

        """ heatmap plot """
        if is_final_test:
            fig, ax = plt.subplots()
            sns.heatmap(cm, ax=ax, annot=True, cmap=plt.cm.Reds, fmt='d',
                        xticklabels=['Truly non-epileptic', 'Truly epileptic'],
                        yticklabels=['Predicted non-epileptic', 'Predicted epileptic'])
            plt.savefig('..\\outputImg\\confusion_matrix\\' + self.restr_name + '.png')

    def evaluation(self, y_test: np.ndarray, Y_hat_test: np.ndarray, is_final_test: bool):
        """
        Metrix evaluation

        :param y_test: y value of test part dataset
        :param Y_hat_test: predicted y value for test part dataset
        :param is_final_test: True if it's the final test, false if it's a development test
        """

        """ printing metrics results """
        print('Final metrics: \n', classification_report(y_test, Y_hat_test))

        """ evaluating metrics """
        accuracy_score_val = accuracy_score(y_test, Y_hat_test)
        balanced_score_val = balanced_accuracy_score(y_test, Y_hat_test)
        precision_score_val = precision_score(y_test, Y_hat_test)
        recall_score_val = recall_score(y_test, Y_hat_test)
        f1_score_val = f1_score(y_test, Y_hat_test)
        roc_curve_val = roc_auc_score(y_test, Y_hat_test)

        """ creating evaluation dataframe image """
        if is_final_test:
            score = [accuracy_score_val, balanced_score_val, precision_score_val, recall_score_val, f1_score_val,
                     roc_curve_val]
            dict = {'Evaluation': self.eval, 'Score on the test set': score}
            df = pd.DataFrame(dict)
            df = df.style.background_gradient(vmin=0.8, vmax=1.0)
            dfi.export(df, '..\\outputImg\\evaluation\\' + self.restr_name + '.png', fontsize=30)

        """ printing metrics evaluations """
        print("Accuracy score on the test set: ", accuracy_score_val)
        print("Balanced Accuracy score on the test set: ", balanced_score_val)
        print("Precision Score on the test set: ", precision_score_val)
        print("Recall score on the test set: ", recall_score_val)
        print("F1 score on the test set: ", f1_score_val)
        print("ROC curve score on the test set: ", roc_curve_val, "\n")

    def create_pmml(self, X: np.ndarray, y: np.ndarray, best_params_: Dict[str, Any]):
        """
        .pmml creation

        :param X: features value to create .pmml
        :param y: output value to create .pmml
        :param best_params_: hyperparameters set by @train_assessment

        ----- NEEDS TO DOWNGRADE sklearn TO 1.2.2!! -----
        """

        """ preparing """
        pipeline = PMMLPipeline([("classifier", self.model_class(**best_params_))])

        """ training """
        pipeline.fit(X, y)

        """ creating .pmml """
        sklearn2pmml(pipeline, "..\\pmml\\" + self.restr_name + ".pmml")
