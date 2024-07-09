# Epileptic Seizure Recognition
![image](https://github.com/bertonfederico/EpilepticSeizureRecognition/assets/105301467/bb796ac9-2b2c-485b-9be0-38a834299020)


Epilepsy is a neurological disorder characterized by a predisposition to
the onset of epileptic (or comitial) seizures. It is one of the most
frequent neurological diseases, with a prevalence of about 1 percent
(500,000 patients) in Italy.

The electroencephalogram makes it possible to investigate brain
function, record electrical activity, and highlight certain
abnormalities, called epileptiforms, which, however, cannot always be
classified as epileptic during visualization.

For this reason, Machine Learning can positively help the classification
of electrical activities: starting from the analysis of different
electrical activities that have already been classified in the past, ML
offers the possibility of unearthing epileptogenic fragments, even those
that by pure human analysis in the past have not been classified as
abnormal.

This project aims to test different Machine Learning models and analyze
the results; in addition, it is possible, via a web page, to insert new
fragments of electrical activity to visualize their classification
(epilepsy or not) by choosing a ML model.

## 🚿 Data import and cleaning

Since the first column (i.e., the one showing the exam number from which
it was obtained) is not needed, it can be removed from the dataset.

``` python
import pandas as pd

""" Loading """
data = pd.read_csv('Dataset/data.csv')

""" Removing first column (not useful) """
data.drop('Unnamed', axis=1, inplace=True)
```

## 👀 Data Observation

The original data from which the dataset was obtained consisted of 500
different EEG recordings, each of which contained 4097 data points and
lasts about 23.6 seconds; each recording was divided into 23 fragments
(each fragment then lasts a little more than a second), resulting in
11500 rows of data. The distance between two points therefore is ***23.6
sec ÷ 4096 = 0.0058 sec***.

Each row is suffixed into 5 types, i.e., the y column contains a value
between {1, 2, 3, 4, 5}:

-   y = 5: normal subjects with eyes open;
-   y = 4: normal subjects with eyes closed;
-   y = 3: seizure-free recordings from patients with epilepsy (opposite the epileptogenic zone)
-   y = 2: seizure-free recordings from patients with epilepsy (epileptogenic zone);
-   y = 1: recordings from patients with epilepsy showing seizure activity.

### y diffences

Let\'s look at the general difference between EEG recordings of type {1,
2, 3, 4, 5}:
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

""" Creating a dataframe with one row for each value of X """
df_splitted_seizure_short = (samples_to_show
                .melt(id_vars=['y'], var_name='time_label', value_name='EEG', ignore_index=False)
                .reset_index()
                .rename(columns={'index': 'id'})
            )

""" Getting time_index column from time_label """
df_splitted_seizure_short['time_label'] = (df_splitted_seizure_short['time_label'].str.translate(str.maketrans('', '', 'X')).astype(int))

""" Creating and showing the graph """
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

![y_differences](https://github.com/bertonfederico/EpilepticSeizureRecognition/assets/105301467/48bb057b-7392-4af7-ad8a-6df678581876)


It is possible to guess that epileptic EEGs manifest larger electrical
activities in height (electrical potential) and width (frequency).

### y-values number

For each value of y is contained exactly 2300 rows:
``` python
data_y_1 = data[data['y'] == 1]
data_y_2 = data[data['y'] == 2]
data_y_3 = data[data['y'] == 3]
data_y_4 = data[data['y'] == 4]
data_y_5 = data[data['y'] == 5]

labels = ('Epileptic area\nin seizure activity', 'Tumor area', 'Healthy area\nin tumor brain',
          'Healthy brain\n- eyes closed', 'Healthy brain\n- eyes open')
colors = plt.cm.Blues(np.linspace(0.2, 0.7, len(labels)))
sizes = [len(data_y_1.index), len(data_y_2.index), len(data_y_3.index), len(data_y_4.index), len(data_y_5.index)]
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors)
plt.show()
```
![row_number](https://github.com/bertonfederico/EpilepticSeizureRecognition/assets/105301467/90836943-4bc9-417f-bde5-08cd9b1690c4)


## 🔧 Correction of y-values

Since the epileptogenic rows are only those with y equal to 1, we modify
the values of y:

-   if y ∈ {2, 3, 4, 5}, it will be transformed into y = 0 (of y =
    \'Non-epileptic\'), i.e., non-epileptogenic rows
-   if y = 1 it will remain so (or y = \'Epileptic\'), i.e.,
    epileptogenic rows
``` python
import numpy as np

data['y'] = np.where(data['y'] == 1, 1, 0)
data_exploratory = data.copy()
data_exploratory['y'] = np.where(data_exploratory['y'] == 1, 'Epileptic', 'Non-epileptic')
```

## 📉 Exploratory Data Analysis: examples

### Correlation matrix
``` python
import seaborn as sns

""" removing y values """
heatmap_data = data_exploratory.iloc[:,0:178]

""" creating heatmap """
sns.heatmap(heatmap_data.corr())
plt.title("Heatmap")
```
![eeg_heatmap](https://github.com/bertonfederico/EpilepticSeizureRecognition/assets/105301467/cff58afb-13e1-49aa-af73-e94e1cc247df)

### EEG potential (*μV*): ***min, max & Standard Deviation*** {#eeg-potential-μv-min-max--standard-deviation}

To analyze how the EEGs used extend, and to visualize the difference
between epileptic and nonepileptic electrical activity, relational,
distribution and categorical plot can be created regarding the
***minimum/maximum/mean*** or ***Standard Deviation*** potential and
frequency of neurological beats.

Let\'s analyze the potential of EEG records; now we start by calculating
the minimum and maximum values:
``` python
""" adding min, max and id columns """
df_potential_min_max = pd.DataFrame()
df_potential_min_max['Min'] = data_exploratory.min(axis=1, numeric_only=True)
df_potential_min_max['Max'] = data_exploratory.max(axis=1, numeric_only=True)
df_potential_min_max['id'] = data_exploratory.index + 1
df_potential_min_max['y'] = data_exploratory['y']
df_potential_min_max = (df_potential_min_max.melt(id_vars=['id', 'y'], var_name='Value type', value_name='value', ignore_index=True))
```

We can try to analyze it with ***categorical plotes***, ***distribution
plot*** with ***Kernel Density Estimation*** and ***relational plot***:
``` python
""" categorical plot """
cat = sns.catplot(
    data=df_potential_min_max,
    kind='boxen',
    x='y',
    y='value',
    col='Value type'
)
cat.set_ylabels('Potential values (μV)', clear_inner=False)
cat.fig.subplots_adjust(top=.9)
cat.fig.suptitle("Categorical plot - min/max potential values")


""" distribution plot with Kernel Density Estimation """
dist = sns.displot(
    data=df_potential_min_max,
    kind='kde',
    x='value',
    hue='y',
    col='Value type'
)
dist.fig.subplots_adjust(top=.9)
dist.fig.suptitle("Kernel Density Estimate - min/max potential values")
dist.set_xlabels('Potential values (μV)', clear_inner=False)


""" relational plot """
rel = sns.relplot(
    data=df_potential_min_max,
    x='id',
    y='value',
    col='y',
    hue='Value type',
    style='Value type',
    edgecolor='#CCFFFF'
)
rel.set_xlabels('Neurological beats recordings', clear_inner=False)
rel.set_ylabels('Potential values (μV)', clear_inner=False)
rel.fig.subplots_adjust(top=.9)
rel.fig.suptitle("Relational plot - min/max potential values")
rel.set(xticklabels=[])

axes = rel.axes.flat[0]
axes.axhline(df_potential_min_max[(df_potential_min_max['y'] == 'Non-epileptic') & (df_potential_min_max['Value type'] == 'Min')]['value'].mean(), ls='--', linewidth=2, color='red')
axes.axhline(df_potential_min_max[(df_potential_min_max['y'] == 'Non-epileptic') & (df_potential_min_max['Value type'] == 'Max')]['value'].mean(), ls='--', linewidth=2, color='red')
axes = rel.axes.flat[1]
axes.axhline(df_potential_min_max[(df_potential_min_max['y'] == 'Epileptic') & (df_potential_min_max['Value type'] == 'Min')]['value'].mean(), ls='--', linewidth=2, color='red')
axes.axhline(df_potential_min_max[(df_potential_min_max['y'] == 'Epileptic') & (df_potential_min_max['Value type'] == 'Max')]['value'].mean(), ls='--', linewidth=2, color='red')
```
![cat_plot](https://github.com/bertonfederico/EpilepticSeizureRecognition/assets/105301467/ea003cbd-8bab-459e-8ebf-087e6f70fd5e)
![kde_plot](https://github.com/bertonfederico/EpilepticSeizureRecognition/assets/105301467/02825f6b-7b17-495b-894a-e62f5a7fbed8)
![rel_plot](https://github.com/bertonfederico/EpilepticSeizureRecognition/assets/105301467/2271bd6b-9467-4a7e-a901-9bfaae359760)

Now let\'s try to analyze the difference between epileptic and
non-epileptic EEG based on the Standard Deviation of the potential:
``` python
""" calculating Standard Deviation """
df_potential_std = data_exploratory.iloc[:, 0:178]
df_potential_std = df_potential_std.std(axis=1).reset_index()
df_potential_std['y'] = data_exploratory['y']
df_potential_std['id'] = df_potential_std.index + 1


""" categorical plot """
cat = sns.catplot(
    data=df_potential_std,
    kind='boxen',
    x='y',
    y=0
)
cat.set_ylabels('Potential values (μV)', clear_inner=False)
cat.fig.subplots_adjust(top=.9)
cat.fig.suptitle("Categorical plot - std potential values")


""" distribution plot with Kernel Density Estimation """
dist = sns.displot(
    data=df_potential_std,
    kind='kde',
    x=0,
    hue='y',
)
dist.fig.subplots_adjust(top=.9)
dist.fig.suptitle("Kernel Density Estimate - std potential values")
dist.set_xlabels('Potential values (μV)', clear_inner=False)


""" relational plot """
rel = sns.relplot(
    data=df_potential_std,
    x='id',
    y=0,
    col='y',
    edgecolor='#CCFFFF'
)
rel.set_xlabels('Neurological beats recordings', clear_inner=False)
rel.set_ylabels('Potential values (μV)', clear_inner=False)
rel.fig.subplots_adjust(top=.9)
rel.fig.suptitle("Relational plot - std potential values")
rel.set(xticklabels=[])
```
![cat_plot](https://github.com/bertonfederico/EpilepticSeizureRecognition/assets/105301467/24267fb2-3fe4-4f50-8309-fd85c2698e17)
![kde_plot](https://github.com/bertonfederico/EpilepticSeizureRecognition/assets/105301467/ecb4bfb3-c0b9-4161-8365-ddaca738643e)
![rel_plot](https://github.com/bertonfederico/EpilepticSeizureRecognition/assets/105301467/aac89d1c-4dc5-4852-908e-d97fdb8916d2)

It is evident how epileptic seizures are manifested in much larger EEGs
from the point of view of electrical potential.

### EEG frequency (*Hz*)

Let us now analyze the frequenca of EEG samples. The celebral waves are
divided according to frequenca into:

-   delta rhythm: 0.5-4 Hz (average amplitude of 150 µV);
-   theta rhythm: 4-7.5 Hz (average amplitude of 75 mV);
-   alpha rhythm: 8-13 Hz (average amplitude of 30 mV);
-   theta-sigma rhythm: 12-14 Hz (average amplitude of 5-50 µV);
-   beta rhythm: 13.5-30 Hz (average amplitude of 19 mV).

For frequency calculation, it is necessary to find the maximum or
minimum values of the sine wave-like trend. Since choosing the analysis
by the minimum or maximum values does not result in any difference,
let\'s try to perform it on the maximum values:
``` python
df_frequency_prepare = data_exploratory.iloc[:, 0:178]
df_frequency_prepare['max_number'] = 0
for count in range(2,178):
    df_frequency_prepare.loc[(data_exploratory["X"+str(count-1)] < data_exploratory["X"+str(count)]) &
                             (data_exploratory["X"+str(count)] > data_exploratory["X"+str(count+1)]),
                    "max_number"] = df_frequency_prepare['max_number'] + 1
```

We now search for the maximum and minimum values for each EEG line and
convert the values to seconds (i.e., knowing that each line contains the
values recorded in one second, we divide the values obtained by 178)

Let\'s now try to compare the frequency of epileptic and nonepileptic
EEGs; we visualize it in categorical plot, distribution plot, and
relational plot:
``` python
""" frequency calculation """
df_frequency_mean = pd.DataFrame()
df_frequency_mean['Freq'] = df_frequency_prepare['max_number']/1.02
df_frequency_mean['id'] = data_exploratory.index + 1
df_frequency_mean['y'] = data_exploratory['y']


""" categorical plot """
cat = sns.catplot(
    data=df_frequency_mean,
    kind='boxen',
    x='y',
    y='Freq'
)
cat.set_ylabels('Frequence values (Hz)', clear_inner=False)
cat.fig.subplots_adjust(top=.9)
cat.fig.suptitle("Categorical plot - frequency values")


""" distribution plot with Kernel Density Estimation """
dist = sns.displot(
    data=df_frequency_mean,
    kind='kde',
    x='Freq',
    hue='y',
)
dist.fig.subplots_adjust(top=.9)
dist.fig.suptitle("Kernel Density Estimate - frequency values")
dist.set_xlabels('Frequence values (Hz)', clear_inner=False)


""" relational plot """
ret = sns.relplot(
    data=df_frequence_mean,
    x='id',
    y='Freq',
    col='y',
    edgecolor='#CCFFFF'
)
ret.set_xlabels('Neurological beats recordings', clear_inner=False)
ret.set_ylabels('Frequence values (Hz)', clear_inner=False)
ret.fig.subplots_adjust(top=.9)
ret.fig.suptitle("Relational plot - frequence values")
ret.set(xticklabels=[])

axes = ret.axes.flat[0]
axes.axhline(df_frequence_mean[(df_frequence_mean['y'] == 'Non-epileptic')]['Freq'].mean(), ls='--', linewidth=2, color='red')
axes = ret.axes.flat[1]
axes.axhline(df_frequence_mean[(df_frequence_mean['y'] == 'Epileptic')]['Freq'].mean(), ls='--', linewidth=2, color='red')
```
![cat_plot](https://github.com/bertonfederico/EpilepticSeizureRecognition/assets/105301467/40dcc664-a582-4f3e-a676-f501fc4226e9)
![kde_plot](https://github.com/bertonfederico/EpilepticSeizureRecognition/assets/105301467/e01bf8fa-10d7-4cda-a3ad-3da5aa5599dd)
![rel_plot](https://github.com/bertonfederico/EpilepticSeizureRecognition/assets/105301467/35a5ab5d-971d-4713-8689-f54dec6d25ca)

As noted by the graphs regarding the width of the periods, we can also
observe this more clearly from the frequency representations: the rhythm
is far less in the case of epileptic EEGs.

## Ⓜ️ Machine Learning Algorithms: example with Neural Network

Several Machine Learning algorithms have been carried out, and those that created more significant results are given below:

-   Support Vector Classification.;
-   Neural Network;

Let\'s visualize ***Neural Network***.

### Preprocessing

First, it is necessary to divide the values of each row of the dataset
into:

-   ***X***, ***y***: ***X*** corresponds to the EEG data, ***y***
    corresponds to the type (epileptic or non-epileptic)
-   ***X_train***, ***y_train***, ***X_development_test***, ***y_development_test***, ***X_test***, ***y_test***:
    - ***_train*** corresponds to 70% of the rows, while
    - ***_development_testn*** corresponds to 10% of the rows, while
    - ***_test*** corresponds to 20%.

Initially, training will be carried out on ***_train*** by cross-validation and testing then on ***_development_test***. Finally, once the functioning of these algorithms is established, the algorithm will be trained on ***_train*** + ***_development_test*** and tested on ***_test***.
``` python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

""" dividing EEG data and epileptic/non-epileptic typology """
X = data.iloc[:,0:178].values
y = data.iloc[:,178].values

""" dividing in training, development and test set """
X_train_dev, X_test, y_train_dev, y_test = train_test_split(X, y, test_size=0.2)
X_train, X_development, y_train, y_development, = train_test_split(X_train_dev, y_train_dev, test_size=1 / 8)

```

### Training & developement test: a bit long 😉

Now treaning can be launched! Let\'s use GridSearchCV to run the training with different gyperparameters and figure out which one allows better prediction:
``` python
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


"""""""""""""""""""""""""""""""""""""""""""""
""""""  Training and development test  """"""
"""""""""""""""""""""""""""""""""""""""""""""
""" scaling """
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_development = sc.transform(X_development)

""" preparing """
test_grid = {
    'hidden_layer_sizes': [(178, 178, 178), (300, 300, 300)],  # N° of layers and neurons
    'activation': ['relu', 'tanh'],                            # Activation functions
    'max_iter': [1000, 100000]                                 # Maximum iterations
}
cross_val = GridSearchCV(self.model_class(), self.test_grid, cv=7, verbose=10)

""" training """
cross_val.fit(X_train, y_train)

""" test """
y_hat_test  = cross_val.predict(X_test)

""" best hyperparameters """
best_params_ = cross_val.best_params_
```

This training determined which hyperparameters are best to use in this case, namely a number of layers equal to 3, each with a number of neurons equal to 300, and an activation of "relu" type. In addition, it was found that the maximum number of iterations that can be performed is preferable to set it at a number that is not too high (1000 or so).

![image](https://github.com/bertonfederico/EpilepticSeizureRecognition/assets/105301467/54cddbda-5b30-44e6-ab8b-0ca89629b9a8)

#### Metrics scores
Now training and developement testing is over! Let\'s look at metrics scores to see if the chosen ML algorithm is suitable:
``` python
from sklearn.metrics import f1_score, accuracy_score, cohen_kappa_score, jaccard_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.metrics import classification_report

accuracy_score_val = accuracy_score(y_vals, y_hat_vals)
balanced_score_val = balanced_accuracy_score(y_vals, y_hat_vals)
precision_score_val = precision_score(y_vals, y_hat_vals)
recall_score_val = recall_score(y_vals, y_hat_vals)
f1_score_val = f1_score(y_vals, y_hat_vals)
roc_curve_val = roc_auc_score(y_vals, y_hat_vals)

""" printing metrics evaluations """
print("Accuracy score on the development test set: ", accuracy_score_val)
print("Balanced Accuracy score on the development test set: ", balanced_score_val)
print("Precision Score on the development test set: ", precision_score_val)
print("Recall score on the development test set: ", recall_score_val)
print("F1 score on the development test set: ", f1_score_val)
print("ROC curve score on the development" test set: ", roc_curve_val)
```
![assessment_test_NeuralNetwork](https://github.com/bertonfederico/EpilepticSeizureRecognition/assets/105301467/a35a6af5-1012-43f9-a17e-ee70e700a531)

The test set scores are between 93% and 98%, so Neural Network is an excellent Machine Learning algorithm for classification of epileptic
EEGs!

#### Confusion matrix

To have a graphic proof of the evaluation just obtained, let\'s visualize a confusion matrix:
``` python
from sklearn.metrics import confusion_matrix


cm = confusion_matrix(y_test, y_hat_test)
fig, ax = plt.subplots()
sns.heatmap(cm, ax = ax, annot = True, cmap = plt.cm.Reds, fmt = 'd', xticklabels = ['Predicted non-epileptic', 'Predicted epileptic'], yticklabels = ['Truly non-epileptic', 'Truly epileptic'])
```
![assessment_test_NeuralNetwork](https://github.com/bertonfederico/EpilepticSeizureRecognition/assets/105301467/39e8dcba-dd3d-4987-b13c-b5c31d9a1ef5)


### Final training & test
Now, once we have established the optimal hyperparameters and learned how reliable this prediction algorithm turns out to be, we can run the training of the algorithm on the dataset of ***_train*** + ***_development*** and run the final test on the ***_test*** dataset.
The results obtained from this final test are shown below:

<div style="display: flex; justify-content: space-around;">
    <img src="https://github.com/bertonfederico/EpilepticSeizureRecognition/assets/105301467/32efd9f4-60f5-4204-afb3-49cfeeb2a9c4" style="width: 50%;">
    <img src="https://github.com/bertonfederico/EpilepticSeizureRecognition/assets/105301467/7fdc27d3-54dd-4777-944b-fa0e2ba02344" style="width: 45%;">
</div>


## ❓ PMML creation: example with Neural Network

### ❗Needs to downgrade SKLEARN to 1.2.2❗

If we want to use what we get from the ML algorithm, it is possible to
use PMML (Predictive Model Markup Language), which is an open XML-based
markup language designed to allow the description of predictive analysis
models that can be shared among different systems and applications.

Having already run the ML algorithm with GridSearchCV we know which
settings provide better results\...but beware! ***sklearn2pmml*** does
not yet accept the latest version of sklearn, for this duindi we have to
downgrade it 😢

First we launch the Openscoring server:
``` python
import subprocess
import warnings
warnings.filterwarnings('ignore')


# starting Openscoring server
subprocess.Popen(["java", "-jar", "Server/lib/openscoring-server-executable-2.1.1.jar"])
```

Then we run the ML algorithm with precise activation functions:
``` python
from sklearn.neural_network import MLPClassifier
from sklearn2pmml.pipeline import PMMLPipeline


# setting grid
grid_pmml = {
    'hidden_layer_sizes': (300, 300, 300),           # N° of layers and neurons
    'activation': 'relu',                            # Activation functions
    'max_iter': 1000                                 # Maximum iterations
}

# preparing
pipeline = PMMLPipeline([
    ('scaler', StandardScaler()),
    ("classifier", self.model_class(**grid_pmml)),
])

# training
pipeline.fit(X, y)
```

Next we create the .pmml format from what was obtained through training
and testing:
``` python
from sklearn2pmml import sklearn2pmml


# creating .pmml
sklearn2pmml(pipeline, "Server\\pmml\\NeuralNetwork.pmml")
```

Finally, we activate the Openscoring client in order to use the .pmml we
just obtained:

``` python
# starting Openscoring with .pmml
subprocess.run(["powershell", "java -cp Server\\lib\\openscoring-client-executable-2.1.1.jar org.openscoring.client.Deployer --model http://localhost:8080/openscoring/model/NeuralNetwork --file Server\\pmml\\NeuralNetwork.pmml"], shell=True)       
```
