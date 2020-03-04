import numpy as np
import matplotlib.pyplot as plt
import math as m
from scipy import signal



sigma = 0.5
X1 = np.zeros(500)
n = np.arange(0, 100, 1)
f0 = 0.1
X2 = np.cos(2 * np.pi * f0 * n)
X3 = np.zeros(300)

# noiseless signal
X = np.concatenate((X1, X2, X3), None)
# noisy signal
X_n = X + np.sqrt(0.5) * np.random.randn(X.size)

# deterministic detector
def get_perc_rate(X):
    res = 0
    for n in range(X.size):
        res += X[n]*m.cos(2*m.pi*f0*n + sigma)
    return res

res = []
for i in range(X.size):
    res.append(get_perc_rate(X_n[i:i+60])/25)



#np.linspace()
#random signal detector
#res = signal.correlate(X_n, X, mode='same')
h = np.exp(-2 * m.pi * 1j * f0 * n)
y = np.abs(np.convolve(h, X_n, 'same'))

# plotting result
fig, ax = plt.subplots(4, 1)
ax[0].plot(X)
ax[1].plot(X_n)
ax[2].plot(res)
ax[3].plot(y)
plt.show()
