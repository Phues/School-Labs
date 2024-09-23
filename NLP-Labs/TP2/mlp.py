import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def init_params(nx, nh, ny):
    W1 = np.random.normal(size=(nh, nx+1))
    W2 = np.random.normal(size=(ny, nh+1))
    
    return W1, W2

#tanh activation function
def tanh(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))

#softmax activation function
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)

#relu activation function
def relu(z):
    return np.maximum(0, z)

#sigmoid activation function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward(params, X):
    W1, W2 = params
    dict = {}
    a0 = np.vstack([np.ones(X.shape[0]), X.T])

    z1 = np.matmul(W1, a0) 
    a1 = tanh(z1)

    a2_ = np.vstack([np.ones(a1.shape[1]), a1])
    z2 = np.matmul(W2, a2_)
    a2 = softmax(z2)

    dict = {'a1': a1, 'z1': z1, 'a2': a2, 'z2': z2}
    return dict

def loss_accuracy(Y, Y_hat):
    m = Y.shape[0]
    J = - (1/m) * np.sum(Y * np.log(Y_hat))
    loss = J
    TP = 0
    TN = 0
    for i in range(Y.shape[1]):
        TP += np.sum(Y[:, i] * Y_hat[:, i])
        TN += np.sum((1 - Y[:, i]) * (1 - Y_hat[:, i]))
    accuracy = (TP + TN) / Y.shape[1]
    return loss, accuracy

def backward(X, params, outputs, Y):
    W1, W2 = params
    outputs['z1'] = np.vstack([np.ones(outputs['z1'].shape[1]), outputs['z1']])
    outputs['a1'] = np.vstack([np.ones(outputs['a1'].shape[1]), outputs['a1']])
    X = np.vstack([np.ones(X.shape[0]), X.T])

    gradients = {}
    gradients["dZ2"] = outputs['a2'] - Y.T

    result = np.matmul(gradients["dZ2"], outputs['a1'].T)
    gradients["dw2"] = result

    relu_derivative = np.where(outputs['z1'][1:] > 0, 1, 0)

    result = np.matmul(W2.T[1:], gradients["dZ2"])
    gradients["dZ1"] = result * relu_derivative

    result = np.matmul(gradients["dZ1"], X.T)
    gradients["dw1"] = result

    return gradients

def sgd(params, grads, eta):
    W1, W2 = params
    W1 -= eta * grads['dw1']
    W2 -= eta * grads['dw2']
    return W1, W2

def train(X, Y, n_batch, n_epochs, eta):
    nx = X.shape[1]
    nh = 16
    ny = Y.shape[1]
    params = init_params(nx, nh, ny)
    loss_hist = []
    accuracy_hist = []
    for i in range(n_epochs):
        for j in range(0, X.shape[0], n_batch):
            X_batch = X[j:j + n_batch]
            Y_batch = Y[j:j + n_batch]
            outputs = forward(params, X_batch)
            loss, accuracy = loss_accuracy(Y_batch, outputs['a2'].T)
            loss_hist.append(loss)
            accuracy_hist.append(accuracy)
            grads = backward(X_batch, params, outputs, Y_batch)
            params = sgd(params, grads, eta)
    return loss_hist, accuracy_hist

def plot_loss_accuracy(loss, accuracy):
    plt.plot(loss)
    plt.plot(accuracy)
    plt.show()

