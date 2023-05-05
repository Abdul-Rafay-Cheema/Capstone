import pandas as pd

data = pd.read_csv('C:\\Users\Abdul Rafay Cheema\\Desktop\\Zoo.csv')

# Understanding and visualizing the dataset to begin with
print(data.head())
print(data.tail())

print(data.shape)

print(data.columns)

print(data.nunique())

print(data.describe())

# Next, we remove all the null and duplicate rows to clean our dataset and start performing our EDA.
print(data.dropna())
print(data.drop_duplicates())

# Now we start performing Exploratory Data Analysis based on our 5 Questions
import matplotlib.pyplot as plt
import seaborn as sns

# Question 1 :Show the trend of domestication with regards to the type pf animals.
sns.scatterplot(x='type', y='domestic', data=data)
plt.show()

# Question 2: For each specific number of legs, how many animals from the dataset posses them?
fig = plt.figure(figsize=(8, 8))
ax = fig.gca()
data.hist(ax=ax, bins=20)
plt.show()

# Question 3: What is the average number of legs present in the dataset?
average = round(data['legs'].mean())
print('The average number of legs amongst all the animals is', average)

# Question 4:	What are the data types of each column (e.g., numerical, categorical, text)?

data_types = data.dtypes
print('Data types of each column:')
print(data_types)

# Question no.5: Are there any missing values or outliers in the dataset?
missing_values = data.isna().sum()
print('Missing values:')
print(missing_values)

plt.figure(figsize=(10, 8))
data.boxplot()
plt.xticks(rotation=90)
plt.show()

import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport

profile = ProfileReport(data, title="Zoo EDA", html={'style': {'full_width': True}})
# profile.to_notebook_iframe()
profile.to_file("EDA")
import webbrowser
import os

chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
url = "file:/" + os.path.realpath("EDA.html")
webbrowser.register('chrome', None, webbrowser.BackgroundBrowser(chrome_path))
webbrowser.get('chrome').open_new_tab(url)

# Now we start our Predictive Data Analysis (PDA)
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from sklearn.exceptions import DataDimensionalityWarning
from sklearn import preprocessing
from sklearn.preprocessing import OrdinalEncoder

for col in data:
    if data[col].dtype == 'object':
        data[col] = OrdinalEncoder().fit_transform(data[col].values.reshape(-1, 1))
data

class_label = data['type']
data = data.drop(['type'], axis=1)
data = (data - data.min()) / (data.max() - data.min())
data['type'] = class_label
data

# pre-processing
zoo_data = data.copy()
le = preprocessing.LabelEncoder()
animal = le.fit_transform(list(zoo_data["animal"]))
hair = le.fit_transform(list(zoo_data["hair"]))
feathers = le.fit_transform(list(zoo_data["feathers"]))
eggs = le.fit_transform(list(zoo_data["eggs"]))
milk = le.fit_transform(list(zoo_data["milk"]))
airborne = le.fit_transform(list(zoo_data["airborne"]))
aquatic = le.fit_transform(list(zoo_data["aquatic"]))
predator = le.fit_transform(list(zoo_data["predator"]))
tooth = le.fit_transform(list(zoo_data["toothed"]))
backbone = le.fit_transform(list(zoo_data["backbone"]))
breathes = le.fit_transform(list(zoo_data["breathes"]))
venomous = le.fit_transform(list(zoo_data["venomous"]))
fins = le.fit_transform(list(zoo_data["fins"]))
legs = le.fit_transform(list(zoo_data["legs"]))
tail = le.fit_transform(list(zoo_data["tail"]))
domestic = le.fit_transform(list(zoo_data["domestic"]))
catsize = le.fit_transform(list(zoo_data["catsize"]))
family = le.fit_transform(list(zoo_data["type"]))

# Modeling and preparation

# Predictive analytics model development by comparing different Scikit-learn classification algorithms
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier

x = list(
    zip(hair, feathers, eggs, milk, airborne, aquatic, predator, tooth, backbone, breathes, venomous, fins, legs, tail,
        domestic, catsize))
y = list(animal)
# Test options and evaluation metric
num_folds = 5
seed = 7
scoring = 'accuracy'
# Model Test/Train
# Splitting what we are trying to predict into 4 different arrays -
# X train is a section of the x array(attributes) and vise versa for Y(features)
# The test data will test the accuracy of the model created
import sklearn.model_selection

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.20, random_state=seed)

np.shape(x_train), np.shape(x_test)
models = []
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
# evaluate each model in turn
results = []
names = []
print("Performance on Training set")
for name, model in models:
    kfold = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    msg += '\n'
    print(msg)

# Compare Algorithms' Performance
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
# Model Evaluation by testing with independent/external test data set.
# Make predictions on validation/test dataset
# Model Evaluation by testing with independent/external test data set.
# Make predictions on validation/test dataset
models.append(('DT', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
models.append(('GBM', GradientBoostingClassifier()))
models.append(('RF', RandomForestClassifier()))
dt = DecisionTreeClassifier()
nb = GaussianNB()
gb = GradientBoostingClassifier()
rf = RandomForestClassifier()
best_model = rf
best_model.fit(x_train, y_train)
y_pred = best_model.predict(x_test)
print("Best Model Accuracy Score on Test Set:", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

# Model Performance Evaluation Metric 2
# Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Model Evaluation Metric 4-prediction report
for x in range(len(y_pred)):
    print("Predicted: ", y_pred[x], "Actual: ", y_test[x], "Data: ", x_test[x], )
