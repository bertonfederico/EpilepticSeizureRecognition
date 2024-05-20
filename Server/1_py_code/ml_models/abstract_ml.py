from typing import Type, Dict, Any

import dataframe_image as dfi
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, balanced_accuracy_score, \
    roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
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
                               X_development: np.ndarray, y_development: np.ndarray, over_sampling: bool):
        """
        Train & test phase for hyperparameters

        :param X_train: features dataset for model training
        :param y_train: y dataset for model training
        :param X_development: features dataset for test prediction
        :param y_development: y dataset for prediction checking
        :param over_sampling: Ture if data are over sampled, false otherwise
        :return: best_params_
        """

        y_hat_train, y_hat_test, best_params_ = self.train_assessment(X_train, y_train, X_development, over_sampling)
        self.confusion_matrix(y_train, y_hat_train, False, False)
        self.confusion_matrix(y_development, y_hat_test, False, True)
        self.evaluation(y_train, y_hat_train, False, "development", False)
        self.evaluation(y_development, y_hat_test, False, "development", True)
        return best_params_

    def train_assessment(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, over_sampling: bool):
        """ refit """
        if over_sampling:
            refit = scoring = 'roc_auc'
        else:
            refit = scoring = 'f1'

        """ preparing """
        cross_val = GridSearchCV(self.model_class(), self.test_grid, cv=7,  scoring=scoring, refit=refit)

        """ training """
        cross_val.fit(X_train, y_train)
        Y_hat_train = cross_val.predict(X_train)

        """ test """
        Y_hat_test = cross_val.predict(X_test)

        """ printing hyperparameters results """
        print('Best hyperparameters: ', cross_val.best_params_)

        return Y_hat_train, Y_hat_test, cross_val.best_params_

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

        y_hat_train, y_hat_test = self.final_train_test(X_train, y_train, X_test, best_params_set)
        self.confusion_matrix(y_train, y_hat_train, True, False)
        self.confusion_matrix(y_test, y_hat_test, True, True)
        self.evaluation(y_train, y_hat_train, True, "final", False)
        self.evaluation(y_test, y_hat_test, True, "final", True)

    def final_train_test(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                         best_params_set: Dict[str, Any]):
        """ preparing """
        new_classifier = self.model_class(**best_params_set)

        """ training """
        new_classifier.fit(X_train, y_train)
        Y_hat_train = new_classifier.predict(X_train)

        """ test """
        Y_hat_test = new_classifier.predict(X_test)

        return Y_hat_train, Y_hat_test

    def confusion_matrix(self, y_vals: np.ndarray, y_hat_vals: np.ndarray, is_final_test: bool, is_test: bool):
        """
        Confusion matrix

        :param y_vals: real y values
        :param y_hat_vals: predicted y values
        :param is_final_test: True if it's the final train/test, false if it's a development train/test
        :param is_test: True if it's test, False if it's development
        """

        actual_dir, print_string = self.get_phase(is_final_test, is_test)
        print('\r\n####### Confusion matrix ' + print_string + ' ######')
        cm = confusion_matrix(y_vals, y_hat_vals)

        """ results """
        print("Non-epileptic classified as non-epileptic: ", cm[0, 0])
        print("Non-epileptic classified as epileptic: ", cm[0, 1])
        print("Epileptic classified as non-epileptic: ", cm[1, 0])
        print("Epileptic classified as epileptic: ", cm[1, 1])

        """ heatmap plot """
        fig, ax = plt.subplots()
        sns.heatmap(cm, ax=ax, annot=True, cmap=sns.color_palette("Blues_d", as_cmap=True), fmt='d',
                    xticklabels=['Predicted non-epileptic', 'Predicted epileptic'],
                    yticklabels=['Truly non-epileptic', 'Truly epileptic'])
        plt.gcf().set_facecolor('none')
        plt.savefig('..\\outputImg\\confusion_matrix\\' + actual_dir + self.restr_name + '.png')

    def evaluation(self, y_vals: np.ndarray, y_hat_vals: np.ndarray, is_final_test: bool, test_type: str, is_test: bool):
        """
        Metrix evaluation

        :param y_vals: real y values
        :param y_hat_vals: predicted y values
        :param is_final_test: True if it's the final test, false if it's a development test
        :param test_type: string indicating development test or final test
        :param is_test: True if it's test, False if it's development
        """

        """ evaluating metrics """
        accuracy_score_val = accuracy_score(y_vals, y_hat_vals)
        balanced_score_val = balanced_accuracy_score(y_vals, y_hat_vals)
        precision_score_val = precision_score(y_vals, y_hat_vals)
        recall_score_val = recall_score(y_vals, y_hat_vals)
        f1_score_val = f1_score(y_vals, y_hat_vals)
        roc_curve_val = roc_auc_score(y_vals, y_hat_vals)

        actual_dir, print_string = self.get_phase(is_final_test, is_test)
        print('\r\n####### Metrics ' + print_string + ' ######')
        score = [accuracy_score_val, balanced_score_val, precision_score_val, recall_score_val, f1_score_val,
                 roc_curve_val]
        dict = {'Evaluation': self.eval, 'Score on the test set': score}
        df = pd.DataFrame(dict)
        df = df.style.background_gradient(vmin=0.88, vmax=1.0)
        dfi.export(df, '..\\outputImg\\evaluation\\' + actual_dir + self.restr_name + '.png', fontsize=30)

        """ printing metrics evaluations """
        print("Accuracy score on the " + test_type + " test set: ", accuracy_score_val)
        print("Balanced Accuracy score on the " + test_type + " test set: ", balanced_score_val)
        print("Precision Score on the " + test_type + " test set: ", precision_score_val)
        print("Recall score on the " + test_type + " test set: ", recall_score_val)
        print("F1 score on the " + test_type + " test set: ", f1_score_val)
        print("ROC curve score on the " + test_type + " test set: ", roc_curve_val)

    @staticmethod
    def get_phase(is_final_test, is_test):
        if is_final_test and is_test:
            actual_dir = "final_test_"
            print_string = " - final test"
        elif is_final_test and not is_test:
            actual_dir = "final_train_"
            print_string = " - final train"
        elif not is_final_test and is_test:
            actual_dir = "assessment_test_"
            print_string = " - assessment test"
        else:
            actual_dir = "assessment_train_"
            print_string = " - assessment train"
        return actual_dir, print_string

    def create_pmml(self, X: np.ndarray, y: np.ndarray, best_params_: Dict[str, Any]):
        """
        .pmml creation

        :param X: features value to create .pmml
        :param y: output value to create .pmml
        :param best_params_: hyperparameters set by @train_assessment

        ----- NEEDS TO DOWNGRADE sklearn TO 1.2.2!! -----
        """

        """ preparing """
        pipeline = PMMLPipeline([
            ('scaler', StandardScaler()),
            ("classifier", self.model_class(**best_params_)),
        ])

        """ training """
        pipeline.fit(X, y)

        """ creating .pmml """
        sklearn2pmml(pipeline, "..\\pmml\\" + self.restr_name + ".pmml")
