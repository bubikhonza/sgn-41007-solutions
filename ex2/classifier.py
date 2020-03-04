import scipy.io
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import metrics
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle

mat = scipy.io.loadmat('twoClassData.mat')
data = mat['X']
labels = mat['y'][0]
data, labels = shuffle(data, labels)

tr_data = data[:200]
te_data = data[200:]
tr_labels = labels[:200]
te_labels = labels[200:]

#KNN
model = KNeighborsClassifier()
model.fit(tr_data, tr_labels)
predicted = model.predict(te_data)
print('KNN: '  + str(accuracy_score(predicted, te_labels) * 100) + '%')

#LDA
model2 = LinearDiscriminantAnalysis()
model2.fit(tr_data, tr_labels)
predicted2 = model2.predict(te_data)
print('LDA: ' + str(accuracy_score(predicted2, te_labels) * 100) + '%')
