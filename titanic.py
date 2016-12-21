import pandas as pds
from sklearn.cross_validation import KFold
from sklearn.linear_model.base import LinearRegression
import numpy as np

data_set = pds.read_csv('train.csv')

AGE = 'Age'

data_set[AGE] = data_set[AGE].fillna(data_set[AGE].median())

GENDER = 'Sex'
data_set.loc[data_set[GENDER] == 'male', GENDER] = 0
data_set.loc[data_set[GENDER] == 'female', GENDER] = 1

EMBARKED = 'Embarked'
data_set[EMBARKED] = data_set[EMBARKED].fillna('S')
data_set.loc[data_set[EMBARKED] == 'S', EMBARKED] = 0
data_set.loc[data_set[EMBARKED] == 'C', EMBARKED] = 1
data_set.loc[data_set[EMBARKED] == 'Q', EMBARKED] = 2
# print(data_set.describe())

algorithm = LinearRegression()

kf = KFold(data_set.shape[0], n_folds=3, random_state=2)

predictors = [GENDER, 'Pclass']
# predictors = [GENDER, 'Age']

predictions = []

for train, test in kf:
    train_predictors = (data_set[predictors].iloc[train, :])

    train_target = data_set['Survived'].iloc[train]
    algorithm.fit(train_predictors, train_target)

    test_prediction = algorithm.predict(data_set[predictors].iloc[test, :])
    predictions.append(test_prediction)

predictions = np.concatenate(predictions, axis=0)

predictions = predictions > 0.5

accuracy = np.sum(data_set['Survived'] == predictions)/data_set.shape[0]

print(accuracy)


