#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 10:44 2018
@author: F.A.Marcus
Original code from Udacity - nd109 Neural Network
 - Adicionando mais um hidden layer.
 - Usando a funcao tanh para ativacao
 - Calculo e grafico de log-error
"""
import numpy as np
import matplotlib.pyplot as plt
from functions_lib import sigmoid, hyp_tan, log_error
from data_UCLA import features, targets, features_test, targets_test

np.random.seed(42)

# Hyperparameters
n_hidden1 = 3  # number of hidden units 1st layer
n_hidden2 = 2  # number of hidden units 2nd layer
epochs =800
learnrate = 0.1

n_records, n_features = features.shape

errors = []
last_loss = None

print("Hidden Layers: (",n_hidden1,n_hidden2,")")
print("Learn rate: ",learnrate)

# Initialize weights
W1 = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden1)) #W1
W2 = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_hidden1,  n_hidden2)) #W2
W3 = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_hidden2,1))

# Training the network!
for e in range(epochs):
    del_W1 = np.zeros(W1.shape)
    del_W2 = np.zeros(W2.shape)
    del_W3 = np.zeros(W3.shape)
    
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        # 1st hidden layer
        h1_in = np.dot(x,W1)      # h = W1.x
        h1_out = hyp_tan(h1_in)   # h = f(W1.x)
        # 2nd hidden layer
        h2_in = np.dot(h1_out,W2)
        h2_out = hyp_tan(h2_in)
        # Output layer
        y_in = np.dot(h2_out, W3)
        y_hat = sigmoid(y_in)     # y^ = f(W3.h2)

        ## Backward pass ##
        # TODO: Calculate error term
        delta3 = (y - y_hat) * sigmoid(y_in,deriv=True) # delta3 =( y-y^) f'(y)
        delta2 = np.dot(W3,delta3) * hyp_tan(h2_in,deriv=True)
        delta1 = np.dot(W2,delta2) * hyp_tan(h1_in, deriv=True) # delta1 = W2.d2 f'(h2)
        
        # TODO: Update the change in weights
        del_W3 += delta3 * h2_out[:, None]
        del_W2 += delta2 * h1_out[:, None]   #dW2 = dW2 + delta2.h1
        del_W1 += delta1 * x[:, None]    #dW1 = dW1 + delta1.x

    # TODO: Update weights
    W1 += learnrate * del_W1 / n_records 
    W2 += learnrate * del_W2 / n_records
    W3 += learnrate * del_W3 / n_records

    # Compute error
    h1_c = hyp_tan(np.dot(x, W1))
    h2_c = hyp_tan(np.dot(h1_c, W2))
    out = sigmoid(np.dot(h2_c,W3))
    loss = np.mean((out - targets) ** 2)
    #loss = np.mean(log_error(y,out))
    errors.append(loss)
    
    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        if last_loss and last_loss < loss:
            print("Epoch:", e, "Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Epoch:", e, "Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
h1_c = hyp_tan(np.dot(x, W1))
h2_c = hyp_tan(np.dot(h1_c, W2))
out = sigmoid(np.dot(h2_c,W3))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test[:, None])
print("Prediction accuracy: {:.4f}".format(accuracy))
print("Min error value:",np.min(errors))
print("Epoch of min error value:",np.argmin(errors))

# Plot error
#plt.title("Error plot")
plt.title("hidden: (" + str(n_hidden1) + " " + str(n_hidden2) + ")" \
          + ", learn_rate: " + str(learnrate) + ", Acc: {:.2f}".format(accuracy))
plt.xlabel("Number of Epochs")
plt.ylabel("Error")
plt.plot(errors)
plt.grid()
plt.show()
