# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=2>

# Pump it up! Modeling work

# <codecell>
from __future__ import print_function

import pandas as pd
import re
from itertools import combinations
import collections

# import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix

import patsy
import numpy as np
import matplotlib.pyplot as plt
from impute import imputeTrain, fillTest
# %matplotlib inline

# TODO: make impute_func work, need workaround on the dataframe splitting

# <codecell>


class PumpModel(object):
    def __init__(self,
                 csv_train_X='../data/train_X.csv',
                 csv_train_y='../data/train_y.csv',
                 csv_test_X='../data/test.csv'):
        """
        input:
            csv_train_X - file location
            csv_train_y - file location
        """
        self.csv_train_X = csv_train_X
        self.csv_train_y = csv_train_y
        self.csv_test_X = csv_test_X

        self.impute_func = None
        self.fill_test_func = None
        self.flag_interactions = False
        self.flag_clean_features = False

        self.df = self.read_n_convert_functional_labels()

        # define mapping of labels<->numeric
        # (ideally this should be a class constant)
        self.LABELS = ['non functional',
                       'functional needs repair',
                       'functional']

        print('PumpModel obj. initialzed. data read into df. \n'
              'PumpModel.run_batch() to see quick analysis results.')

    def run_batch(self,
                  flag_interactions=False,
                  flag_clean_features=False,
                  impute_func=None,
                  fill_test_func=None):

        if impute_func:
            self.impute_func = impute_func
            self.fill_test_func = fill_test_func

        self.flag_interactions = flag_interactions
        self.flag_clean_features = flag_clean_features

        print('reading and transforming data...')
        print('split train and test, testing models...')
        self.run_models()
        # it looks RandomForest does us better, let's try a new parameter
        print('\ntrying RandomForestClassifier with n_estimators=200...')
        model = RandomForestClassifier(n_estimators=200)
        self.split_n_fit_train(model)

        # let's try another new parameter
        print('\ntrying RandomForestClassifier with n_estimators=300...')
        model = RandomForestClassifier(n_estimators=300)
        self.split_n_fit_train(model)

        # also use KFold to make sure we cross validate
        print('\ntrying RandomForestClassifier with n_estimators=200 '
              'using KFold...')
        model = RandomForestClassifier(n_estimators=200)
        self.run_KFold(model)

        # so far, we only used the automatic feature selection.
        # To get some idea of what feature really matters, we can check
        # the importance of features
        print('\nsort and barplot features...')
        self.sort_feature_imporances()

    def run_batch_realtest(self,
                           model=RandomForestClassifier(n_estimators=200),
                           flag_interactions=False,
                           flag_clean_features=False,
                           impute_func=None,
                           fill_test_func=None):
        """ take real test data and generate submission file
        input:
            model=RandomForestClassifier(n_estimators=200),
            flag_interactions=False,
            flag_clean_features=False,
            impute_func=None,
            fill_test_func=None

        return:
            self.y_pred_realtest
        """
        df = self.df
        if impute_func:
            print('imputing data...')
            df, self.df_X_realtest = self.impute_data(df, self.df_X_realtest,
                                                      impute_func,
                                                      fill_test_func)

        print('get X, y from training set')
        (self.X, self.y) = self.ready_for_model_train(
                            df, flag_interactions=flag_interactions,
                            flag_clean_features=flag_clean_features)

        print('fitting model...')
        model.fit(self.X, self.y)

        print('preparing X, y from test set...')
        X_test, y_test = self.ready_for_model_test(
            self.df_X_realtest, flag_interactions)

        print('predicting y...')
        self.y_pred_realtest = model.predict(X_test)
        self.print_test_predictions(self.y_pred_realtest)
        return self.y_pred_realtest

    def run_models(self, df=pd.DataFrame(),
                   models=[LogisticRegression(), DecisionTreeClassifier(),
                           KNeighborsClassifier(), GaussianNB(),
                           RandomForestClassifier()]):
        """ run split_n_fit through severl models, default:
                [LogisticRegression(), DecisionTreeClassifier(),
                           KNeighborsClassifier(), GaussianNB(),
                           RandomForestClassifier()]
            pre-run required: PumpModel.ready_for_model
        """
        if df.empty:
            df = self.df

        for model in models:
            self.split_n_fit_train(model, df,
                                   self.impute_func, self.fill_test_func)

    def run_KFold(self, model, df=pd.DataFrame(), n_folds=5,
                  impute_func=None, fill_test_func=None):
        """ given model, X, y, print score of the fit on KFold test"""
        if df.empty:
            df = self.df
        scores = []
        cnt = 0
        for train_index, test_index in KFold(df.shape[0], n_folds=n_folds):
            df_train = df.iloc[train_index, :]
            df_test = df.iloc[test_index, :]
            if impute_func:
                df_train, df_test = self.impute_data(df_train, df_test,
                                                     impute_func,
                                                     fill_test_func)
            X_train, y_train = self.ready_for_model_train(
                df_train, flag_clean_features=self.flag_clean_features,
                flag_interactions=self.flag_interactions)

            X_test, y_test = self.ready_for_model_test(
                df_test, flag_interactions=self.flag_interactions)

            model.fit(X_train, y_train.ravel())
            score = model.score(X_test, y_test)
            scores.append(score)
            cnt += 1
            print('fold', cnt, ':', score)

        print('avg score:', np.mean(scores))

        # keep a copy of the train, test set
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model_fitted = model
        self.show_confusion_matrix()

    def read_n_convert_functional_labels(self):
        """
            1. read training data sets X and y
            2. merge these two data sets
            3. convert functional to 2, non-functional to 0,
            functional-needs-repaird to 1, new column name 'status'
            return: dataframe df
        """

        df_X = pd.read_csv(self.csv_train_X)
        df_X = df_X.fillna('nan_value')
        df_y = pd.read_csv(self.csv_train_y)
        self.df_X_realtest = pd.read_csv(self.csv_test_X)
        self.df_X_realtest = self.df_X_realtest.fillna('nan_value')
        self.realtest_IDs = self.df_X_realtest.id
        df = pd.merge(df_y, df_X, how='left', on='id')
        df.loc[:, 'status'] = df.status_group
        df.loc[df.status.str.startswith('functional needs'), 'status'] = 1
        df.loc[df.status_group.str.startswith('non'), 'status'] = 0
        df.loc[~df.status.str.startswith('functional').isnull(), 'status'] = 2
        df.status = df.status.astype(int)
        return df

    def ready_for_model_train(self,
                              df=pd.DataFrame(),
                              flag_interactions=False,
                              flag_clean_features=False):
        """
        process data

        input:
            flag_iteractions - True to add columns of additional feature
                                interactions
            flag_clean_features - True to keep most common N categories in
                                the features with too many of them

        create/modify:
            self.cols_continuous
            self.cols_categorical
            self.cols_giveup
            self.df_X
            self.df_y
            self.X
            self.y
            self.cols_X
            self.r_formula
        """
        if df.empty:
            df = self.df
        N = 25
        if flag_clean_features:
            df = self.clean_features(df, N=N)

        # seperate features to continuous, categorical
        self.cols_continuous, self.cols_categorical, self.cols_giveup = \
            self.preprocess_features(df, N=N)

        # remove the labels
        for to_remove in ['id', 'status', 'status_group']:
            try:
                self.cols_continuous.remove(to_remove)
            except ValueError:
                pass
            try:
                self.cols_categorical.remove(to_remove)
            except ValueError:
                pass

        # convert df to X, y by patsy
        self.r_formula = ('status ~' +
                          ' + '.join(self.cols_continuous) +
                          ' + C(' + ') + C('.join(self.cols_categorical) + ')'
                          )
        self.df_y, self.df_X = patsy.dmatrices(self.r_formula,
                                               data=df,
                                               return_type='dataframe')

        # include interactions of features if flagged
        if flag_interactions:
            self.df_X = self.interactions(self.df_X)

        self.cols_X = self.df_X.columns
        self.X = self.df_X.values
        self.y = self.df_y.values
        return (self.X, self.y)

    def ready_for_model_test(self, df_test,
                             flag_interactions=False):
        """
        process test data, matching columns with self.df_X_test

        input:
            df_test - test DataFrame
            flag_iteractions - True to add columns of additional feature
                                interactions

        create/modify:
            self.df_X_test
            self.df_y_test
            self.X_test
            self.y_test

        return:
            self.X_test
            self.y_test
        """

        # add column status to test as place holder for real test set
        if 'status' not in df_test.columns:
            df_test.loc[:, 'status'] = 0

        self.df_y_test, self.df_X_test = patsy.dmatrices(
                                               self.r_formula,
                                               data=df_test,
                                               return_type='dataframe')
        # matching columns with df_X
        self.df_X_test = self.match_cols(self.df_X, self.df_X_test)

        if flag_interactions:
            self.df_X_test = self.interactions(self.df_X_test)

        assert (self.df_X.columns == self.df_X_test.columns).all()

        # include interactions of features if flagged
        if flag_interactions:
            self.df_X_test = self.interactions(self.df_X_test)

        self.X_test = self.df_X_test.values
        self.y_test = self.df_y_test.values
        return (self.X_test, self.y_test)

    def match_cols(self, df_X_0=pd.DataFrame(), df_X_1=pd.DataFrame()):
        """ match df_X_1 columns with df_X_0.
        will remove extra columns and add missing ones"""

        if df_X_0.empty:
            df_X_0 = self.df_X
        if df_X_1.empty:
            df_X_1 = self.df_X_realtest

        cols_0 = set(df_X_0.columns)
        cols_1 = set(df_X_1.columns)
        to_drop = cols_1 - cols_0
        to_add = cols_0 - cols_1
        df_X_1 = df_X_1.drop(to_drop, axis=1)
        for c in to_add:
            df_X_1.loc[:, c] = 0.
        return df_X_1[df_X_0.columns]

    def clean_features(self, df=pd.DataFrame(), N=25,
                       feat_list=['scheme_name']):
        """
        Keep the most common entries for some features
        default for now just include scheme_name since it seems to
        be particularly important
        """
        print('cleaning features...')
        if df.empty:
            df = self.df

        for feat in feat_list:
            least_common = [x[0] for x in collections.Counter(df[feat]).
                            most_common()[N:-1]]
            for label in least_common:
                df.loc[:, feat] = df[feat].replace(label, 'other')
        return df

    def preprocess_features(self, df, N=25):
        """ pre-process,  collect columns based type (continuous, categorical,
        giveup (too many categories))
            input:
                N - maximum num of categories accepted in one feature
            return:
                cols_continuos - list
                cols_categorical - list
                cols_giveup - list
        """

        self.cols_continuous = []
        self.cols_categorical = []
        self.cols_giveup = []

        for c in df:
            if c.endswith('_code'):
                self.cols_categorical.append(c)
            elif df[c].dtype in [int, float]:
                self.cols_continuous.append(c)
            elif df[c].dtype == object:
                if df[c].nunique() < N + 3:
                    self.cols_categorical.append(c)
                else:
                    self.cols_giveup.append(c)
        return (self.cols_continuous, self.cols_categorical, self.cols_giveup)

    def interactions(self, df_X):
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

    def impute_data(self, df_train, df_test, impute_func, fill_test_func):
        if not impute_func or not fill_test_func:
            raise Exception('need to input both '
                            'impute_func and fill_test_func')
        print('imputing data...')
        self.df_train_imp, self.impute_map = impute_func(df_train)
        self.df_test_imp = fill_test_func(df_test, self.impute_map)
        return self.df_train_imp, self.df_test_imp

    def split_df(self, df, impute_func=None, fill_test_func=None):
        df_train, df_test = train_test_split(df, random_state=42)
        if impute_func:
            df_train, df_test = self.impute_data(df_train, df_test,
                                                 impute_func,
                                                 fill_test_func)
        return df_train, df_test

    def split_n_fit_train(self, model, df=pd.DataFrame(),
                          impute_func=None,
                          fill_test_func=None):
        """ given model, df_X, df_y, print score of the fit on test """

        if df.empty:
            df = self.df

        self.df_train, self.df_test = self.split_df(df,
                                                    impute_func,
                                                    fill_test_func)

        X_train, y_train = self.ready_for_model_train(
                        self.df_train,
                        flag_interactions=self.flag_interactions,
                        flag_clean_features=self.flag_clean_features)
        X_test, y_test = self.ready_for_model_test(
            self.df_test, flag_interactions=self.flag_interactions)
        model.fit(X_train, y_train.ravel())
        print()
        print('{}'.format(model).split('(')[0])
        print(model.score(X_test, y_test))
        self.X_test = X_test
        self.y_test = y_test
        self.model_fitted = model
        self.show_confusion_matrix()

    def fit(self, model):
        X = self.X
        y = self.y
        model.fit(X, y.ravel())
        self.model_fitted = model

    def sort_feature_imporances(self, model_fitted=None):
        """ input:
                fitted_model: model has to be ensemble or tree type and fitted
                cols_X: list of features
            print:
                features ranked by importance
            plot:
                the first 20 important features
        """

        # in tree and ensemble type of models, we can use feature_importances_
        if not model_fitted:
            model_fitted = self.model_fitted
        cols_X = self.df_X.columns
        importances = model_fitted.feature_importances_
        std = np.std([tree.feature_importances_
                      for tree in model_fitted.estimators_],
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

    def show_confusion_matrix(self):
        print('confusion matrix:')
        y_pred = self.model_fitted.predict(self.X_test)
        y_actual = self.y_test

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
        print("recall TP / (TP + FN)", recall)
        precision = TP * 1. / (TP + FP)
        print("precision TP / (TP + FP)", precision)

    def print_test_predictions(self,
                               y_pred,
                               output_file='../data/test_predict_y.csv'):
        """
        Saves predictions on test set to csv file.

        input:
            self
            y_pred - predicted values for test set (0,1,2)
            output_file - filename to save predictions to after mapping
                          to proper labels

        Flow for generating submission:
           pm = PumpModel()
           [y_pred,impute_map] = pm.run_batch_realtest()
           pm.print_test_predictions(y_pred)

           # output is in test_predict_y.csv

        """

        print("Printing results")

        # convert prediction numbers to corresponding labels
        labels = self.LABELS
        y_pred = [labels[int(x)] for x in y_pred]

        # get corresponding test ids
        test_ids = self.realtest_IDs

        # zip together ids and results and print
        results_df = pd.DataFrame(zip(test_ids, y_pred),
                                  columns=['id', 'status_group'])
        results_df.to_csv(output_file, index_label=False, index=False)

    def gen_test_set(self,
                     csv_train_X='../data_test/train_X.csv',
                     csv_train_y='../data_test/train_y.csv',
                     csv_test_X='../data_test/test.csv',
                     csv_test_y='../data_test/test_y_dummy.csv'):
        """
        Generates random test/training split from current training set
            and writes to files.
        input:
             self
           filenames to write to
             csv_train_X='../data_test/train_X.csv',
             csv_train_y='../data_test/train_y.csv',
             csv_test_X='../data_test/test.csv',
             csv_test_y='../data_test/test_y_dummy.csv'):
        """
        print("Generating test set")
        df = self.df
        df_y = df[['id', 'status_group']]
        df_X = df.drop(['status_group'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(df_X, df_y)

        X_train.to_csv(csv_train_X, index_label=False, index=False)
        y_train.to_csv(csv_train_y, index_label=False, index=False)
        X_test.to_csv(csv_test_X, index_label=False, index=False)
        y_test.to_csv(csv_test_y, index_label=False, index=False)


# <codecell>
def main():
    print('run batch data manipulations and test on models')
# <codecell>
    pm = PumpModel()

# <codecell>
    # first batch
    pm.run_batch()

# <codecell>
    # second batch with clean_features
    pm.run_batch(flag_interactions=False, flag_clean_features=True)

# <codecell>
    # 3rd batch with feature interactions
    pm.run_batch(flag_interactions=True, flag_clean_features=False)

# <codecell>
    # 4th batch with feature interactions and clean_features
    pm.run_batch(flag_interactions=True, flag_clean_features=True)

# <codecell>
# if __name__ == "__main__":
#     main()
# <codecell>
