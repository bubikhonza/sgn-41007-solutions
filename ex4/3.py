"""
Tampere University
SGN 41007 - Pattern Recognition and Machine Learning (Autumn 2019)
Exercise 4 Solution - Question 3 Template

Contact:    oguzhan.gencoglu@tut.fi (Office: TE406)
            andrei.cramariuc@tut.fi (Office: TE314)
            rostislav.duda@tuni.fi  (Office: SG307)
"""

# load required packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def log_loss(w, X, y):
    """ 
    Computes the log-loss function at w. The 
    computation uses the data in X with
    corresponding labels in y. 
    """

    L = 0  # Accumulate loss terms here.

    # TODO: Sum up the loss for each sample in X to L
    for n in range(len(y)):
        L = L + np.log(1+np.exp(-y[n]*np.transpose(w)*X[n]))

    return L


def grad(w, X, y):
    """ 
    Computes the gradient of the log-loss function
    at w. The computation uses the data in X with
    corresponding labels in y. 
    """

    G = 0  # Accumulate gradient here.

    # TODO: Sum up the gradient for each sample in X to G
    for n in range(len(y)):
        G = G + (-y[n] * X[n] * np.log(1 + np.exp( -y[n] * np.transpose(w) * X[n]))
                 )/(1 + np.log(1 + np.exp(-y[n] * np.transpose(w) * X[n])))
    return G


if __name__ == "__main__":

    # TODO: Add your code here:

    # 1) Load X and y data:
    X = np.loadtxt("log_loss_data/X.csv", delimiter=',')
    y = np.loadtxt("log_loss_data/y.csv", delimiter=',')

    # 2) Initialize w at random:
    w = np.random.rand(2)

    # 3) Set step_size to a small positive value
    step_size = 0.01

    # 4) Initialize empty lists for storing the path and
    # accuracies:
    W, accuracies = [], []

    for iteration in range(100):
        # 5) Apply the gradient descent rule:
        w = w - step_size*grad(w, X, y)
        W.append(w)
        # 6) Print the current state:
        print(iteration)
        # 7) Compute the accuracy:
        y_pred = np.dot(X, w)
        yhat = (-1)**(y_pred < 0)
        accuracy = np.mean(yhat == y)

        accuracies.append(accuracy)
    # 8) Below is a template for plotting. Feel free to
    # rewrite if you prefer different style:
    # print(W)
    plt.figure(figsize=[5, 5])
    plt.subplot(211)
    plt.plot(np.array(W)[:, 0], np.array(W)[:, 1], 'ro-')
    plt.xlabel('w$_0$')
    plt.ylabel('w$_1$')
    plt.title('Optimization path')

    plt.subplot(212)
    plt.plot(100.0 * np.array(accuracies), linewidth=2)
    plt.ylabel('Accuracy / %')
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.savefig("log_loss_minimization.pdf", bbox_inches="tight")
    plt.show()
