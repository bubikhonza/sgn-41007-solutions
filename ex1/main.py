from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib.image import imread
import cv2 as cv2
import numpy as np
from numpy.linalg import matrix_power
import math as m

# ---------------------3---------------------
'''
mat = loadmat("twoClassData.mat")
print(mat.keys())
X = mat["X"]
y = mat["y"].ravel()
plt.scatter(X[:, 0], X[:, 1], c=y, marker="o", picker=True)
plt.show()
'''
# ---------------------4---------------------
'''
# Read the data
img = imread("uneven_illumination.jpg")
X, Y = np.meshgrid(range(1300), range(1030))
Z = img

x = X.ravel()
y = Y.ravel() 
z = Z.ravel().transpose() #brightness

val = 1300*1030
H = np.zeros((val, 10))
c = np.zeros((10, 1), dtype=np.float)
for i in range(val):
    H[i] = [1, x[i], y[i], x[i]**2, x[i]*y[i], y[i]**2, x[i]**3, x[i]**2*y[i], x[i]*y[i]**2, y[i]**3]

c = (matrix_power((H.transpose() @ H), -1)) @ H.transpose() @ z

theta = c.transpose()
z_pred = H @ theta
Z_pred = np.reshape(z_pred, X.shape)
S = Z - Z_pred
plt.imshow(S, cmap = 'gray')
plt.show()
'''
# ---------------------5---------------------
sample_num = 100
sinusoids_num = 1
f0 = 0.015

#generating noise
x = np.zeros((sample_num, sinusoids_num))
for i in range(sample_num):
    w = np.sqrt(0.3) * np.random.randn(sinusoids_num)
    x[i] = m.sin(2 * m.pi * f0 * i) + w
plt.plot(x, 'b.')


scores = []
frequencies = []
for f in np.linspace(0, 0.5, 1000):
    e = np.ones((1, sample_num))
    n = np.arange(sample_num)
    z = -2*m.pi*1j*f*n
    e = np.exp(z)
    score = abs(np.dot(x.ravel(), e))
    scores.append(score)
    frequencies.append(f)
fHat = frequencies[np.argmax(scores)]
print(fHat)
x_predicted = np.zeros((sample_num, sinusoids_num))
x_actual = np.zeros((sample_num, sinusoids_num))

#ploting
for i in range(sample_num):
    x_predicted[i] = m.sin(2 * m.pi * fHat * i)
    x_actual[i] = m.sin(2 * m.pi * f0 * i)
plt.plot(x_predicted, 'r')
plt.plot(x_actual, 'y')

plt.show()
