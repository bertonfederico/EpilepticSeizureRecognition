import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from data_analysis import data_observation, exploratory_data_analysis

from ml_models.svc import Svc
from ml_models.neural_network import NeuralNetwork

import warnings

warnings.filterwarnings('ignore')

ml_models = [Svc, NeuralNetwork]

""""""""""""""""""""""""""""""""""""""""""""""""
"""  Loading and arrangement of the dataset  """
""""""""""""""""""""""""""""""""""""""""""""""""
""" Loading """
data = pd.read_csv('../../Dataset/data.csv')

""" Removing first column (not useful) """
data.drop('Unnamed', axis=1, inplace=True)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""    Observation of differences between y=1, y=2, y=3, y=4, y=5    """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
data_observation.observation(data)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""             Combining y regarding non-epileptic EEGs             """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""" y = 1 ==> epileptic    ---     y = 0 ==> non-epileptic """
data['y'] = np.where(data['y'] == 1, 1, 0)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                    Exploratory Data Analysis                     """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
exploratory_data_analysis.create_plots(data)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                   Splitting features and output                  """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
X = data.iloc[:, 0:178].values
y = data.iloc[:, 178].values

""" dividing in training, development and test set """
X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_development, y_train, y_development, = train_test_split(X_train_dev, y_train_dev, test_size=1 / 8)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                   Training & assessment phase                    """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('\n\n\n\n###############################################################################################')
print('################################# Training & development phase ################################')
print('###############################################################################################')

""" scaling """
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_development = sc.transform(X_development)
best_params_list = []

""" running training and assessment """
for i, model in enumerate(ml_models):
    act_model = model(i == (len(ml_models) - 1))
    best_params = act_model.train_assessment_phase(X_train, y_train, X_development, y_development)
    best_params_list.append(best_params)


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                           Test phase                             """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
print('\n\n\n\n###############################################################################################')
print('########################################## Test phase #########################################')
print('###############################################################################################')
""" scaling """
sc = StandardScaler()
X_train_dev = sc.fit_transform(X_train_dev)
X_test = sc.transform(X_test)

""" running test """
for i, model in enumerate(ml_models):
    act_model = model(i == (len(ml_models) - 1))
    act_model.final_train_test_phase(X_train_dev, y_train_dev, X_test, y_test, best_params_list[i])
    act_model.create_pmml(X_train_dev, y_train_dev, best_params_list[i])
