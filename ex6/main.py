from scipy.io import loadmat
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

mat = loadmat('./arcene.mat')

X_test = mat['X_test']
X_train = mat['X_train']
y_train = mat['y_train']
y_test = mat['y_test']

model = RandomForestClassifier(n_estimators=100, random_state=0)
model.fit(X_train, y_train)
importances = model.feature_importances_
print(importances)