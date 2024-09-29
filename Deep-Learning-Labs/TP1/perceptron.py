import numpy as np


def train_perceptron(X, y, w, b, r, L=1000):
    for i in range(L):
        for j in range(X.shape[0]):
            error = y[j] - g(np.dot(X[j], w) + b)
            w += error * X[j] * r
            b += error * r
    return w, b

def g(x):
    return 1 if x > 0 else 0


x = np.array([[3,2, 1], [1,1,1], [1, 2, 3]])
y = np.array([0, 1, 1])

#random weights
w = np.random.rand(3)
b = np.random.rand(1)

w, b = train_perceptron(x, y, w, b, .1, 1000)
print(w, b)

#generate random linearly separable data
data = np.random.rand(100, 2)
labels = np.zeros(100)
for i in range(100):
    if data[i, 0] + data[i, 1] > 1:
        labels[i] = 1

#add some overlaping data
data = np.concatenate([data, np.random.rand(20, 2)])
labels = np.concatenate([labels, np.ones(20)])



w = np.random.rand(2)
b = np.random.rand(1)

w, b = train_perceptron(data, labels, w, b, .1, 1000)

#visualize the data if linearly separable

import matplotlib.pyplot as plt
plt.scatter(data[:,0], data[:,1], c=labels)
plt.plot([0, 1], [-b/w[1], (-b-w[0])/w[1]])
plt.show()
            
