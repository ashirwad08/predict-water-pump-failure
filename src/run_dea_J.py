# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=2>

# Pump it up! Modeling work

# <codecell>
from __future__ import print_function

import pandas as pd
import re
from itertools import combinations

# import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, AdaBoostClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix

import collections
import patsy
import numpy as np
import matplotlib.pyplot as plt
from impute import imputeTrain
# %matplotlib inline

# TODO: make impute_func work, need workaround on the dataframe splitting

# <codecell>
def read_n_convert_functional_labels(csv_train_X='../data/train_X.csv',
                                     csv_train_y='../data/train_y.csv',
                                     impute_func=None
                                     ):
    """
        input:
            csv_train_X - file location
            csv_train_y - file location

        1. read training data sets X and y
        2. merge these two data sets
        3. convert functional to 2, non-functional to 0,
           functional-needs-repaird to 1, new column name 'status'

        return: dataframe df
    """

    df_X = pd.read_csv(csv_train_X)
    if impute_func:
        df_X = imputeTrain(df_X)
    df_y = pd.read_csv(csv_train_y)
    df = pd.merge(df_y, df_X, how='left', on='id')
    df.loc[:, 'status'] = df.status_group
    df.replace( {'non functional': 1,
                 'functional needs repair': 2,
                 'functional': 3}, inplace=True)
    df['Log_population'] = np.log(df.population.values + 1)
    
    df['population_zero'] = (df.population < 2) * 1.0
    
    df.drop('population', axis=1, inplace=True)
    #df.drop(['water_quality', 'source', 'scheme_management',
    #         'extraction_type','waterpoint_type_group'], axis=1, inplace=True)
    
    # df.loc[df.status.str.startswith('functional needs'), 'status'] = 1
    # df.loc[df.status_group.str.startswith('non'), 'status'] = 0
    # df.loc[~df.status.str.startswith('functional').isnull(), 'status'] = 2
    # df.status = df.status.astype(int)

    return df


# <codecell>

def ready_for_model(df, flag_interactions=False):
    """
    input:
        df - dataframe
        flag_iteractions - True if additional columns of feature interactions
    return:

    """

    
    
    # Keep the most common entries for some features
    # for now just include scheme_name since it seems to be particularly important
    N = 20
    #feat_list = ['funder','installer','scheme_name']
    feat_list = ['scheme_name']

    for feat in feat_list:
        least_common = [x[0] for x in collections.Counter(df[feat]).most_common()[N:-1]]
        for label in least_common:
            df[feat].replace(label,'NaN',inplace=True)


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

    # include interactions of features if flagged
    if flag_interactions:
        df_X = interactions(df_X)

    cols_X = df_X.columns
    X = df_X.values
    y = df_y.values
    
    return (X, y, cols_X, r_formula, cols_keep, cols_giveup)


# <codecell>
def interactions(df_X):
    """ input: df_X including the transposed dummy columns
        return: df_X with feature interactions
    """
    cols_X = df_X.columns
    cols_X = cols_X.drop('Intercept')
    cols_X_combo = list(combinations(cols_X, 2))

    combo_keep = []
    for c1, c2 in cols_X_combo:
        # ignore dummies X dummies interactions
        if re.search("\[T\.", c1) and re.search("\[T\.", c2):
            continue
        combo_keep.append([c1, c2])

    for c1, c2 in combo_keep:
        df_X[c1 + '_X_' + c2] = df_X[c1] * df_X[c2]

    return df_X


# <codecell>
def split_n_fit(model, X, y):
    """ given model, X, y, print score of the fit on test """
    X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(),
                                                        random_state=42)
    model.fit(X_train, y_train)
    print()
    print('{}'.format(model).split('(')[0])
    print(model.score(X_test, y_test))


# <codecell>
def run_models(X, y):
    for model in [LogisticRegressionCV(n_jobs=-1,solver='lbfgs'), DecisionTreeClassifier(),
                  KNeighborsClassifier(), ExtraTreesClassifier(),
                  AdaBoostClassifier(), RandomForestClassifier()]:
        split_n_fit(model, X, y)


# <codecell>
def sort_feature_imporances(fitted_model, cols_X):
    """ input:
            fitted_model: model has to be ensemble or tree type and fitted
            cols_X: list of features
        print:
            features ranked by importance
        plot:
            the first 20 important features
    """

    # in tree and ensemble type of models, we can use feature_importances_
    model = fitted_model
    importances = model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.estimators_],
                 axis=0)
    indices = importances.argsort()[::-1]
    print("Features ranked by importance:")
    for i, (feature, importance) in enumerate(zip(cols_X[indices],
                                                  importances[indices])):
        print(i, feature, importance)

    # just plot the first 20
    n_first = 20
    plt.figure(figsize=[15, 10])
    plt.bar(range(n_first), importances[indices[0:20]], color='g',
            yerr=std[indices[0:20]], align='center')
    plt.xticks(range(n_first), cols_X[indices[0:20]], rotation=60)
    plt.tight_layout()
    plt.show()


def binarize_y_confustion_matrix(y_actual, y_pred):
    # get y_pred and y_actual

    # convert y's to binary

    # we want True means not-functional or need-repair
    # (where the govern. needs to send people) , False: functional
    # In our data, 2: functional 1: need repair 0: non-functional
    y_pred_bin = (y_pred < 2)
    y_actual_bin = (y_actual < 2)

    conf = confusion_matrix(y_actual_bin, y_pred_bin)

    TP = conf[0, 0]
    FN = conf[0, 1]
    FP = conf[1, 0]
    TN = conf[1, 1]
    recall = TP * 1. / (TP + FN)
    print("recall", recall)
    precision = TP * 1. / (TP + FP)
    print("precision", precision)


# <codecell>

def run_KFold(model, X, y, n_folds=5):
    """ given model, X, y, print score of the fit on KFold test"""
    y = y.ravel()
    scores = []
    cnt = 0
    for train_index, test_index in KFold(len(y), n_folds=n_folds):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
        cnt += 1
        print('fold', cnt, ':', score)

    print('avg score:', np.mean(scores))

    print('confusion matrix:')
    y_pred = model.predict(X_test)
    y_actual = y_test
    binarize_y_confustion_matrix(y_actual, y_pred)
    
def best_RandomForest(X, y):
    from sklearn.grid_search import RandomizedSearchCV
    from scipy.stats import randint as sp_randint
    from sklearn.metrics import accuracy_score
    
    X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(),
                                                        random_state=42)
    
    clf = ExtraTreesClassifier(bootstrap = False)

    grid = {'n_estimators': sp_randint(250, 400),
            'min_samples_leaf' : sp_randint(1, 12),
            'max_features' : sp_randint(5, 50),
            'max_depth': sp_randint(3,30)}

    clf_rfc = RandomizedSearchCV(clf, n_jobs=4, n_iter=10,
                                 param_distributions=grid,
                                 scoring='accuracy')

    y_hat = clf_rfc.fit(X_train,
                    y_train.ravel()).predict(X_test)

    print('Best Params: \n')
    for k, v in clf_rfc.best_params_.items():                                            
        print(k, v)

    print("Accuracy with Random Forest = %4.4f"  %
          accuracy_score(y_test.ravel(), y_hat))
    binarize_y_confustion_matrix(y_test.ravel(), y_hat)
    return(clf_rfc.best_params_)

def best_ExtraTree(X, y):
    from sklearn.grid_search import RandomizedSearchCV
    from scipy.stats import randint as sp_randint
    from sklearn.metrics import accuracy_score
    
    X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(),
                                                        random_state=42)
    
    clf = ExtraTreesClassifier(max_depth=None,
                                 bootstrap = False)

    grid = {'n_estimators': sp_randint(250, 400),
            'min_samples_leaf' : sp_randint(1, 12),
            'max_features' : sp_randint(5, 50)}

    clf_rfc = RandomizedSearchCV(clf, n_jobs=4, n_iter=10,
                                 param_distributions=grid,
                                 scoring='accuracy')

    y_hat = clf_rfc.fit(X_train,
                    y_train.ravel()).predict(X_test)

    print('Best Params: \n', clf_rfc.best_params_ )
    print("Accuracy with Extra Forest = %4.4f"  %
          accuracy_score(y_test.ravel(), y_hat))
    binarize_y_confustion_matrix(y_test.ravel(), y_hat)
    return(clf_rfc.best_params_) 

def best_BoostTree(X, y):
    from sklearn.grid_search import RandomizedSearchCV
    from scipy.stats import randint as sp_randint
    from sklearn.metrics import accuracy_score
    
    X_train, X_test, y_train, y_test = train_test_split(X, y.ravel(),
                                                        random_state=42)
    
    clf = AdaBoostClassifier()

    grid = {'n_estimators': sp_randint(50, 400)}

    clf_rfc = RandomizedSearchCV(clf, n_jobs=4, n_iter=10,
                                 param_distributions=grid,
                                 scoring='accuracy')

    y_hat = clf_rfc.fit(X_train,
                    y_train.ravel()).predict(X_test)

    print('Best Params: \n', clf_rfc.best_params_ )
    print("Accuracy with AdaBoost Forest = %4.4f"  %
          accuracy_score(y_test.ravel(), y_hat))
    binarize_y_confustion_matrix(y_test.ravel(), y_hat)

# <codecell>

def features_opt(X, y, cut=0.90):
    from sklearn.metrics import metrics
    
    clf_tree = DecisionTreeClassifier(criterion='entropy',
            max_depth=40)
    clf_tree.fit(X, y.ravel())

    importances = clf_tree.feature_importances_

    indices = importances.argsort()[::-1]

    cols_ = list(X.columns)
    s = 0
    i = 0
    cols_main = []
    while s < cut:
        s += importances[indices[i]]
        cols_main.append([cols_[i], importances[indices[i]]])
        i += 1

    return(cols_main)

# <codecell>

def main():

    print('reading and transforming data...')
    df = read_n_convert_functional_labels()
    X, y, cols_X, r_formula, cols_keep, cols_giveup = ready_for_model(df, False)

    print('split train and test, testing models...')
    run_models(X, y)

    # it looks RandomForest does us better, let's try a new parameter
    print('\ntrying RandomForestClassifier with n_estimators=200...')
    model = RandomForestClassifier(n_estimators=200)
    split_n_fit(model, X, y)

    # let's try another new parameter
    print('\ntrying RandomForestClassifier with n_estimators=300...')
    model = RandomForestClassifier(n_estimators=300)
    split_n_fit(model, X, y)

    # also use KFold to make sure we cross validate
    print('\ntrying RandomForestClassifier with n_estimators=200 using KFold...')
    model = RandomForestClassifier(n_estimators=200)
    run_KFold(model, X, y)

    # also use KFold to make sure we cross validate
    print('\ntrying RandomForestClassifier with best param and randomazed search')
    best = best_RandomForest(X, y)
    print()
    
    # also use KFold to make sure we cross validate
    model = RandomForestClassifier(max_depth=best['max_depth'],
                                   bootstrap = False,
                                   min_samples_leaf=best['min_samples_leaf'],
                                   max_features=best['max_features'],
                                   n_estimators= best['n_estimators'])
    
    print('\nTrying RandomForestClassifier with')
    for k, v in best.items():
        print(k, v)
    print('using KFold...')
    run_KFold(model, X, y)
    
    # also use KFold to make sure we cross validate
    print('\ntrying ExtraTreeClassifier with best param and randomazed search')
    #best_ExtraTree(X, y)
    #print()
   
    # also use KFold to make sure we cross validate
    #print('\ntrying AdaBoostClassifier with best param and randomazed search')
    #best_BoostTree(X, y)
    #print()
   
    

    
    # so far, we only used the automatic feature selection.
    # To get some idea of what feature really matters, we can check
    # the importance of features
    #print('Main features')

    

if __name__ == "__main__":
    main()
