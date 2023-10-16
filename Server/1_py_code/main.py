import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_analysis import data_observation, exploratory_data_analysis

from ml_models.svc import Svc
from ml_models.neural_network import NeuralNetwork
from ml_models.gaussian_naive_bayes import Gaussian
from ml_models.k_neighbors_classifier import KNeighborsClass
from ml_models.decision_tree_classifier import DecisionTreeClass
from ml_models.decision_tree_regressor import DecisionTreeReg
from ml_models.extra_tree_classifier import ExtraTreeClass
from ml_models.extra_tree_regressor import ExtraTreeReg
ml_models = [NeuralNetwork]#Svc, KNeighborsClass, NeuralNetwork, Gaussian, DecisionTreeClass, DecisionTreeReg, ExtraTreeClass, ExtraTreeReg]




##############################################
### Loading and arrangement of the dataset ###
##############################################
# Loading
data = pd.read_csv('Dataset/data.csv')
# Removing first column (not useful)
data.drop('Unnamed', axis=1, inplace=True)



##################################################################
### Observation of differences between y=1, y=2, y=3, y=4, y=5 ###
##################################################################
#data_observation.observation(data)



##################################################
### Combining the y regarding non-epileptic EEGs ###
##################################################
# y = 1 ==> epileptic, else non-epileptic
data['y'] = np.where(data['y'] == 1, 1, 0)



#################################
### Exploratory Data Analysis ###
#################################
#exploratory_data_analysis.create_plots(data)



#################################
######### Preprocessing #########
#################################
# dividing EEG data and epileptic/non-epileptic typology
X = data.iloc[:,0:178].values
y = data.iloc[:,178].values

# dividing in training set and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



#################################
####### Machine Learning ########
#################################
for i, model in enumerate(ml_models):
    act_model = model(X_train, X_test, y_train, y_test, (i == (len(ml_models) - 1)))
    act_model.run_model()
    act_model.create_pmml(X, y)