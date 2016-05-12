# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=2>

# Pump it up! Modeling work

# <codecell>

import numpy as np

# <codecell>

# use pandas to read in data
import pandas as pd

df_X = pd.read_csv('../data/train_X.csv')
df_y = pd.read_csv('../data/train_y.csv')
df = pd.merge(df_y, df_X, on='id')

# -- convert functional to 2, non-functional to 0, functional-needs-repaird to 1

df.loc[:, 'status'] = df.status_group
df.loc[df.status.str.startswith('functional needs'), 'status'] = 1
df.loc[df.status_group.str.startswith('non'), 'status'] = 0
df.loc[~df.status.str.startswith('functional').isnull(), 'status'] = 2
df.status = df.status.astype(int)

# -- show df.corr()
df.corr()

# <markdowncell>

# the highest abs|correlation| is gps_height for the continuous variable
# it does not say much

# <markdowncell>

# Initial exploration on the data

# <codecell>

# import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
import patsy

# <markdowncell>

# I am having my first trial by just removing features that have too many
# dummies

# <codecell>

df.head()

# <codecell>

def ready_for_model(df):
    cols = list(df.columns.values)

    # keep columns
    cols_keep = []
    cols_giveup = []
    for c in df:
        if df[c].dtype in [int, float]:
            cols_keep.append(c)
        elif df[c].dtype == object:
            if df[c].nunique() < 25:
                cols_keep.append(c)
            else:
                cols_giveup.append(c)

    # remove the labels
    for to_remove in ['id', 'status', 'status_group']:
        cols_keep.remove(to_remove)

    # convert df to X, y by patsy
    r_formula = 'status ~' + ' + '.join(cols_keep)
    df_y, df_X = patsy.dmatrices(r_formula, df, return_type='dataframe')

    cols_X = df_X.columns
    X = df_X.values
    y = df_y.values
    return (X, y, cols_X, r_formula, cols_keep, cols_giveup)

X, y, cols_X, r_formula, cols_keep, cols_giveup = ready_for_model(df)

# <codecell>

r_formula

# <codecell>

# we gave up these features
cols_giveup

# <codecell>

y

# <codecell>

X

# <codecell>

def split_n_fit(model, X, y):
    """ given model, X, y, print score of the fit on test """
    X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(), random_state=42)
    model.fit(X_train, y_train)
    print
    print '{}'.format(model).split('(')[0]
    print model.score(X_test, y_test)


for model in [LogisticRegression(), DecisionTreeClassifier(),
              KNeighborsClassifier(), GaussianNB(), RandomForestClassifier()]:
    split_n_fit(model, X, y)

# <codecell>

# it looks RandomForest does us better, let's try a new parameter

model = RandomForestClassifier(n_estimators=200)
split_n_fit(model, X, y)

# <codecell>

# it looks RandomForest does us better, let's try a new parameter

model = RandomForestClassifier(n_estimators=300)
split_n_fit(model, X, y)

# <codecell>

from sklearn.cross_validation import KFold

model = RandomForestClassifier(n_estimators=200)
y = y.ravel()
for train_index, test_index in KFold(len(y), n_folds=5):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    model.fit(X_train, y_train)
    print model.score(X_test, y_test)

# <codecell>

from sklearn.metrics import confusion_matrix
# get y_pred and y_actual
y_pred = model.predict(X_test)
y_actual = y_test

# convert y's to binary

# y_pred_bin = ~y_pred.astype(bool)  
# y_actual_bin = ~y_actual.astype(bool)  #

# we want True means not-functional or need-repair (where the govern. needs to send people) , False: functional
# In our data, 2: functional 1: need repair 0: non-functional
y_pred_bin = (y_pred < 2)
y_actual_bin = (y_actual < 2)

conf = confusion_matrix(y_actual_bin, y_pred_bin)
conf

# <codecell>

TP = conf[0, 0]
FN = conf[0, 1]
FP = conf[1, 0]
TN = conf[1, 1]
recall = TP * 1./ (TP + FN)
print "recall", recall
precision = TP * 1./ (TP + FP)
print "precision", precision

# <markdowncell>

# so far, we only used the automatic feature selection.
# To get some idea of what feature really matters, we can check
# the importance of features

# <codecell>

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# in tree and ensemble type of models, we can use feature_importances_
importances = model.feature_importances_
std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)
indices = importances.argsort()[::-1]
print "Features ranked by importance:"
for i, (feature, importance) in enumerate(zip(cols_X[indices], importances[indices])):
    print i, feature, importance

# <codecell>

importances.sum()

# <codecell>

# just plot the first 20
n_first = 20
plt.figure(figsize=[15, 10])
plt.bar(range(n_first), importances[indices[0:20]], color='g',
        yerr=std[indices[0:20]], align='center')
plt.xticks(range(n_first), cols_X[indices[0:20]], rotation=60)
plt.tight_layout()
plt.show()

# <markdowncell>

# ### Now, let's do some imputation
# 
# for population, we'll first try to fill in missing value with the mean of groups

# <codecell>

# it is copied from Ash's code
# firstly, replace 
dat = df.copy()
dat.population.replace(to_replace={0: np.nan}, inplace=True)

# <codecell>

#let's look at population by imputing with mean population in neighboring areas, in this order:
#sub-village > ward > lga > region_code

dat.population.fillna(dat.groupby(['subvillage'])['population'].transform('mean'), inplace=True)
dat.population.fillna(dat.groupby(['ward'])['population'].transform('mean'), inplace=True)
dat.population.fillna(dat.groupby(['lga'])['population'].transform('mean'), inplace=True)
dat.population.fillna(dat.groupby(['region_code'])['population'].transform('mean'), inplace=True)

# <codecell>

X, y, cols_X, r_formula, cols_keep, cols_giveup = ready_for_model(dat)

# <codecell>

for model in [LogisticRegression(), DecisionTreeClassifier(),
              KNeighborsClassifier(), GaussianNB(), RandomForestClassifier()]:
    split_n_fit(model, X, y)

# <codecell>


# <codecell>

# it looks RandomForest does us better, let's try a new parameter

model = RandomForestClassifier(n_estimators=200)
split_n_fit(model, X, y)

# <codecell>

# it looks RandomForest does us better, let's try a new parameter

model = RandomForestClassifier(n_estimators=300)
split_n_fit(model, X, y)

# <codecell>

## LOCATION
# gps_height

(dat.gps_height == 0).sum()/float(len(dat.gps_height))

# <codecell>

dat = df.copy()
dat.gps_height.replace(to_replace={0: np.nan}, inplace=True)
#let's look at gps_height by imputing with mean gps_height in neighboring areas, in this order:
#sub-village > ward > lga > region_code

dat.gps_height.fillna(dat.groupby(['subvillage'])['gps_height'].transform('mean'), inplace=True)
dat.gps_height.fillna(dat.groupby(['ward'])['gps_height'].transform('mean'), inplace=True)
dat.gps_height.fillna(dat.groupby(['lga'])['gps_height'].transform('mean'), inplace=True)
dat.gps_height.fillna(dat.groupby(['region_code'])['gps_height'].transform('mean'), inplace=True)

# <codecell>

X, y, cols_X, r_formula, cols_keep, cols_giveup = ready_for_model(dat)

# <codecell>

for model in [LogisticRegression(), DecisionTreeClassifier(),
              KNeighborsClassifier(), GaussianNB(), RandomForestClassifier()]:
    split_n_fit(model, X, y)

# <codecell>

# it looks RandomForest does us better, let's try a new parameter

model = RandomForestClassifier(n_estimators=200)
split_n_fit(model, X, y)

# <codecell>


