from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, GradientBoostingClassifier

'''
digits = load_digits()
#digits.images => 1797 x 8 x 8
#digits.data => 1797 x 64

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2)
#train_test_split => random_state

classifiers = [KNeighborsClassifier(), LinearDiscriminantAnalysis(), SVC(), LogisticRegression()]
for c in classifiers:
    c.fit(X_train, y_train)
    res = c.predict(X_test)
    print("Accuracy for "+ str(c.__class__) + " is: "+  str(accuracy_score(y_test, res)))


import traffic_signs as ts

X, y = ts.load_data('./')
F = ts.extract_lbp_features(X)

classifiers = [KNeighborsClassifier(), LinearDiscriminantAnalysis(),
               SVC(), LogisticRegression()]
for c in classifiers:
    c.fit(F, y)
    res = cross_val_score(c, F, y, cv=5)
    print("Accuracy for "+ str(c.__class__) + " is: "+  str(res.mean()))

'''
import traffic_signs as ts
X, y = ts.load_data('./')
F = ts.extract_lbp_features(X)

classifiers = [RandomForestClassifier(n_estimators=100), ExtraTreesClassifier(n_estimators=100),
               AdaBoostClassifier(n_estimators=100), GradientBoostingClassifier(n_estimators=100)]
for c in classifiers:
    c.fit(F, y)
    res = cross_val_score(c, F, y, cv=5)
    print("Accuracy for "+ str(c.__class__) + " is: "+  str(res.mean()))
