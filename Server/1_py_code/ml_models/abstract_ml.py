from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, jaccard_score, precision_score, recall_score, balanced_accuracy_score
import pandas as pd
import dataframe_image as dfi
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
import subprocess




class AbstractMl(object):

    #################################
    ########### variables ###########
    #################################
    model_name = None
    model_class = None
    restr_name = None
    eval_name = None
    X_train = None
    X_test = None
    y_train = None
    y_test = None
    Y_hat_test = None
    grid = None
    grid_pmml = None
    is_last = None

    eval = ['Accuracy:', 'F1:', 'Cohen Kappa:', 'Jaccard:', 'Precision:', 'Recall:', 'Balanced Accuracy:']
    dict = {'Evaluation': eval}
    total_df = pd.DataFrame(dict)




    #################################
    ############## init #############
    #################################
    def __init__(self, X_train, X_test, y_train, y_test, is_last):
        print('\n\n################################# ' + self.model_name + ' #################################')
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.is_last = is_last
        self.restr_name = self.model_name.replace(" ", "")
        self.grid = {}



    #################################
    ########### Run model ###########
    #################################
    def run_model(self):
        self.train_test()
        self.evaluation()
        self.confusion_matrix()



    #################################
    ####### Training and Test #######
    #################################
    def train_test(self):
        #preparing
        cross_val = GridSearchCV(self.model_class(), self.grid, cv = 7, n_jobs=2, verbose=10)

        # training
        cross_val.fit(self.X_train, self.y_train)

        # test
        self.Y_hat_test = cross_val.predict(self.X_test)

        # printing results
        print('Tuned hpyerparameters (best parameters): ', cross_val.best_params_)
        print('Estimator that was chosen by the search: ', cross_val.best_estimator_)
        print('Model classification report with GridSearcg CV: \n', classification_report(self.y_test, self.Y_hat_test))



    #################################
    ########## Evaluation ###########
    #################################
    def evaluation(self):
        # creating evaluation dataframe image
        accuracy_score_val = accuracy_score(self.y_test, self.Y_hat_test)
        f1_score_val = f1_score(self.y_test, self.Y_hat_test)
        cohen_score_val = cohen_kappa_score(self.y_test, self.Y_hat_test)
        jaccard_score_val = jaccard_score(self.y_test, self.Y_hat_test)
        precision_score_val = precision_score(self.y_test, self.Y_hat_test)
        recall_score_val = recall_score(self.y_test, self.Y_hat_test)
        balanced_score_val = balanced_accuracy_score(self.y_test, self.Y_hat_test)
        score = [accuracy_score_val, f1_score_val, cohen_score_val, jaccard_score_val, precision_score_val, recall_score_val, balanced_score_val]
        dict = {'Evaluation': self.eval, 'Score on the test set': score}
        df = pd.DataFrame(dict)
        df = df.style.background_gradient()
        dfi.export(df, 'Server\\outputImg\\evaluation\\' + self.restr_name + '.png', fontsize = 30)

        # printing evaluation
        print("Accuracy score on the test set: ", accuracy_score_val)
        print("F1 score on the test set: ", f1_score_val)
        print("Cohen Kappa Score on the test set: ", cohen_score_val)
        print("Jaccard Score on the test set: ", jaccard_score_val)
        print("Precision Score on the test set: ", precision_score_val)
        print("Recall Score on the test set: ", recall_score_val)
        print("Balanced Accuracy Score on the test set: ", balanced_score_val, "\n")

        # adding to total dataframe
        self.total_df[self.eval_name] = score
        if (self.is_last):
            self.total_df = self.total_df.style.background_gradient()
            dfi.export(self.total_df, 'Server\\outputImg\\evaluation\\total_ml.png', fontsize = 30)



    #################################
    ########## Evaluation ###########
    #################################
    def confusion_matrix(self):
        cm = confusion_matrix(self.y_test, self.Y_hat_test)

        # results
        print("Non-epileptic classified as non-epileptic: ", cm[0, 0])
        print("Non-epileptic classified as epileptic: ", cm[0, 1])
        print("Epileptic classified as non-epileptic: ", cm[1, 0])
        print("Epileptic classified as epileptic: ", cm[1, 1])

        # heatmap plot
        fig, ax = plt.subplots()
        sns.heatmap(cm, ax = ax, annot = True, cmap = plt.cm.Reds, fmt = 'd', xticklabels = ['Non-epileptic', 'Epileptic'], yticklabels = ['Non-epileptic', 'Epileptic'])
        plt.savefig('Server\\outputImg\\confusion_matrix\\' + self.restr_name + '.png')







    #################################
    ######### PMML CREATION #########
    #################################
    ###### NEEDS TO DOWNGRADE sklearn TO 1.2.2!! ######
    def create_pmml(self, X, y):

        # preparing
        pipeline = PMMLPipeline([("classifier", self.model_class(**self.grid_pmml))])

        # training
        pipeline.fit(X, y)

        # creating .pmml
        sklearn2pmml(pipeline, "Server\\pmml\\" + self.restr_name + ".pmml")

        # starting Openscoring with .pmml
        subprocess.run(["powershell", "java -cp C:\\Users\\feder\\Documents\\UniPgDevelop\\EpilepticSeizureRecognition\\Server\\lib\\openscoring-client-executable-2.1.1.jar org.openscoring.client.Deployer --model http://localhost:8080/openscoring/model/" + self.restr_name + " --file C:\\Users\\feder\\Documents\\UniPgDevelop\\EpilepticSeizureRecognition\\Server\\pmml\\" + self.restr_name + ".pmml"], shell=True)        