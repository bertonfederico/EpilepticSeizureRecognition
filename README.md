# 🏥 Epileptic Seizure Recognition
## 🚿 Data import and cleaning



``` python
import pandas as pd

# Loading
data = pd.read_csv('Dataset/data.csv')

# Removing first column (not useful)
data.drop('Unnamed', axis=1, inplace=True)
```



## 👀 Data Observation
### y diffences
``` python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
    
n = 3
data_y_1 = data[data['y'] == 1][:n]
data_y_2 = data[data['y'] == 2][:n]
data_y_3 = data[data['y'] == 3][:n]
data_y_4 = data[data['y'] == 4][:n]
data_y_5 = data[data['y'] == 5][:n]
samples_to_show = pd.concat([data_y_1, data_y_2, data_y_3, data_y_4, data_y_5], axis=0, ignore_index=True)

# Creating a dataframe with one row for each value of X
df_splitted_seizure_short = (samples_to_show
                .melt(id_vars=['y'], var_name='time_label', value_name='EEG', ignore_index=False)
                .reset_index()
                .rename(columns={'index': 'id'})
            )

# Getting time_index column from time_label
df_splitted_seizure_short['time_label'] = (df_splitted_seizure_short['time_label'].str.translate(str.maketrans('', '', 'X')).astype(int))

# Creating and showing the graph
g = sns.relplot(
    data=df_splitted_seizure_short,
    kind='line',
    x='time_label',
    y='EEG',
    col='y'
)
g.fig.subplots_adjust(top=.9, left=.07)
g.fig.suptitle("y differences")
g.fig.set_size_inches(13, 5)
plt.legend([], [], frameon=False)
plt.show()
```

### y numbers
``` python
# Checking the number of rows for each value of y
data_y_1 = data[data['y'] == 1]
data_y_2 = data[data['y'] == 2]
data_y_3 = data[data['y'] == 3]
data_y_4 = data[data['y'] == 4]
data_y_5 = data[data['y'] == 5]


labels = 'y = 1', 'y = 2', 'y = 3', 'y = 4', 'y = 5'
sizes = [len(data_y_1.index), len(data_y_2.index), len(data_y_3.index), len(data_y_4.index), len(data_y_5.index)]
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%')
fig.suptitle("Number of rows for each value of y")
plt.show()
```

## 📉 Exploratory Data Analysis: examples {#-exploratory-data-analysis-examples}
### Heatmap
``` python
import seaborn as sns
import numpy as np

data['y'] = np.where(data['y'] == 1, 1, 0)
sns.heatmap(data.corr())
plt.title("Heatmap")
plt.show()
```

### Box plot
``` python
data['y'] = np.where(data['y'] == 1, 'Epileptic', 'Non-epileptic')
df_splitted_seizure = (data
                .melt(id_vars=['y'], var_name='time_label', value_name='EEG', ignore_index=False)
                .reset_index()
                .rename(columns={'index': 'id'})
            )
df_splitted_seizure['time_label'] = (df_splitted_seizure['time_label'].str.translate(str.maketrans('', '', 'X')).astype(int))
sns.catplot(
    data=df_splitted_seizure.groupby(["id", "y"]).std().reset_index(),
    kind='box',
    x='y',
    y='EEG',
).fig.suptitle("Standard deviation")
plt.show()
```

### Kernel Density Estimate
``` python
sns.displot(
    data=df_splitted_seizure.groupby(["id", "y"]).std().reset_index(),
    kind='kde',
    x='EEG',
    hue='y'
).fig.suptitle("Kernel Density Estimate")
plt.show()
```


### EEG Altitude
``` python
altitude = data.iloc[:, 0:177]

altitude['min'] = data.min(axis=1, numeric_only=True)
altitude['max'] = data.max(axis=1, numeric_only=True)

altitude['id'] = altitude.index + 1
altitude['y'] = data['y']
altitude= altitude[['id', 'min', 'max', 'y']]

df_min_max = (altitude.melt(id_vars=['id', 'y'], var_name='min_or_max', value_name='min_max_value', ignore_index=False))

ret = sns.relplot(
    data=df_min_max,
    x='id',
    y='min_max_value',
    col='y',
    hue='min_or_max',
    style='min_or_max',
    edgecolor='#CCFFFF'
)

ret.set_xlabels("Neurological beats recording", clear_inner=False)
ret.set_ylabels('Extreme min/max altitude' + " value", clear_inner=False)
ret.fig.subplots_adjust(top=.9)
ret.fig.suptitle("Extreme min/max altitude value for each Neurological beats recording")
ret.set(xticklabels=[])

axes = ret.axes.flat[0]
axes.axhline(df_min_max[(df_min_max['y'] == 'Non-epileptic') & (df_min_max['min_or_max'] == 'min')]['min_max_value'].mean(), ls='--', linewidth=2, color='red')

axes = ret.axes.flat[1]
axes.axhline(df_min_max[(df_min_max['y'] == 'Epileptic') & (df_min_max['min_or_max'] == 'min')]['min_max_value'].mean(), ls='--', linewidth=2, color='red')

axes = ret.axes.flat[0]
axes.axhline(df_min_max[(df_min_max['y'] == 'Non-epileptic') & (df_min_max['min_or_max'] == 'max')]['min_max_value'].mean(), ls='--', linewidth=2, color='red')

axes = ret.axes.flat[1]
axes.axhline(df_min_max[(df_min_max['y'] == 'Epileptic') & (df_min_max['min_or_max'] == 'max')]['min_max_value'].mean(), ls='--', linewidth=2, color='red')

plt.show()
```

## Ⓜ️ Machine Learning Algorithms: example with Neural Network

``` python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data['y'] = np.where(data['y'] == 'Epileptic', 1, 0)


# dividing EEG data and epileptic/non-epileptic typology
X = data.iloc[:,0:178].values
y = data.iloc[:,178].values

# dividing in training set and test set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```


``` python
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, jaccard_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



#################################
####### Training and Test #######
#################################
#preparing
grid = {
    'hidden_layer_sizes': [(20,), (100, 100)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd'],
    'alpha': [0.0001, 0.05],
    'max_iter': [1000]
}
cross_val = GridSearchCV(MLPClassifier(), grid, cv = 7, n_jobs=2, verbose=10)

# training
cross_val.fit(X_train, y_train)

# test
Y_hat_test = cross_val.predict(X_test)

# printing results
print('Tuned hpyerparameters (best parameters): ', cross_val.best_params_)
print('Estimator that was chosen by the search: ', cross_val.best_estimator_)
print('Model classification report with GridSearcg CV: \n', classification_report(y_test, Y_hat_test))
```
                   precision    recall  f1-score   support

               0       0.97      0.99      0.98      1832
               1       0.97      0.88      0.92       468

        accuracy                           0.97      2300
       macro avg       0.97      0.94      0.95      2300
    weighted avg       0.97      0.97      0.97      2300



``` python
#################################
########## Evaluation ###########
#################################
accuracy_score_val = accuracy_score(y_test, Y_hat_test)
f1_score_val = f1_score(y_test, Y_hat_test)
cohen_score_val = cohen_kappa_score(y_test, Y_hat_test)
jaccard_score_val = jaccard_score(y_test, Y_hat_test)
precision_score_val = precision_score(y_test, Y_hat_test)
recall_score_val = recall_score(y_test, Y_hat_test)
balanced_score_val = balanced_accuracy_score(y_test, Y_hat_test)
eval = ['Accuracy:', 'F1:', 'Cohen Kappa:', 'Jaccard:', 'Precision:', 'Recall:', 'Balanced Accuracy:']
score = [accuracy_score_val, f1_score_val, cohen_score_val, jaccard_score_val, precision_score_val, recall_score_val, balanced_score_val]
dict = {'Evaluation': eval, 'Score on the test set': score}
df = pd.DataFrame(dict)
df = df.style.background_gradient()
df
```

``` python
#################################
####### Confusion matrix ########
#################################
cm = confusion_matrix(y_test, Y_hat_test)
fig, ax = plt.subplots()
sns.heatmap(cm, ax = ax, annot = True, cmap = plt.cm.Reds, fmt = 'd', xticklabels = ['Non-epileptic', 'Epileptic'], yticklabels = ['Non-epileptic', 'Epileptic'])
```

## ❓ PMML creation: example with Neural Network
``` python
###### NEEDS TO DOWNGRADE sklearn TO 1.2.2!! ######


from sklearn2pmml import sklearn2pmml
from sklearn2pmml.pipeline import PMMLPipeline
import subprocess


grid_pmml = {
    'hidden_layer_sizes': (100, 100),
    'activation': 'tanh',
    'solver': 'sgd',
    'alpha': 0.0001,
    'max_iter': 1000
}

# preparing
pipeline = PMMLPipeline([("classifier", MLPClassifier(**grid_pmml))])

# training
pipeline.fit(X, y)

# creating .pmml
sklearn2pmml(pipeline, "Server\\pmml\\neural_network.pmml")

# starting Openscoring with .pmml
subprocess.run(["powershell", "java -cp ...Server\\pmml\\neural_network.pmml"], shell=True)       
```

