# <markdowncell>
# ## Pump it up! Modeling work

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
import patsy

# <markdowncell>
# I am having my first trial by just removing features that have too many
# dummies

# <codecell>
cols = list(df.columns.values)

# keep columns
cols_keep = []
for c in df:
    if df[c].dtype in [int, float]:
        cols_keep.append(c)
    elif df[c].dtype == object:
        if df[c].nunique() < 20:
            cols_keep.append(c)

# remove the labels
for to_remove in ['id', 'status', 'status_group']:
    cols_keep.remove(to_remove)

# convert df to X, y by patsy
r_formula = 'status_group ~' + ' + '.join(cols_keep)
y, X = patsy.dmatrices(r_formula, df)


# <codecell>

def split_n_fit(model, X, y):
    """ given model, X, y, print score of the fit on test """
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    model.fit(X_train, y_train)
    print
    print '{}'.format(model).split('(')[0]
    print model.score(X_test, y_test)


for model in [LogisticRegression(), DecisionTreeClassifier(),
              KNeighborsClassifier(), GaussianNB(), RandomForestClassifier()]:
    split_n_fit(model, X, y)
