import numpy as np
import urllib.request
from sklearn import preprocessing, metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.svm.classes import SVC
from sklearn.tree import DecisionTreeClassifier

url = "http://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"

raw_data = urllib.request.urlopen(url)

dataset = np.loadtxt(raw_data, delimiter=',')

X = dataset[:, 0:8]
y = dataset[:, 8]

normalized_X = preprocessing.normalize(X)

standartized_X = preprocessing.scale(X)

model = ExtraTreesClassifier()

model.fit(X, y)

print(model.feature_importances_)

model = LogisticRegression()

model.fit(X, y)

print(model)

expected = y

predicted = model.predict(X)

print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))


model = GaussianNB()

model.fit(X, y)

expected = y

predicted = model.predict(X)

print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))

model = KNeighborsClassifier()

model.fit(X, y)

expected = y

predicted = model.predict(X)

print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))


model = DecisionTreeClassifier()

model.fit(X, y)

expected = y

predicted = model.predict(X)

print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))

model = SVC()

model.fit(X, y)

expected = y

predicted = model.predict(X)

print(metrics.classification_report(expected, predicted))

print(metrics.confusion_matrix(expected, predicted))