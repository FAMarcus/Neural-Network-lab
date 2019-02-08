#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 18:27:18 2018
@author: F.A.Marcus
Modified code from Udacity - nd109 Neural Network
"""
import numpy as np
from data_UCLA import features, targets, features_test, targets_test

def sigmoid(x,deriv=False):
    """ Calculate sigmoid and its derivative
    """
    if (deriv==True):
       return x*(1-x)
    return 1/(1+np.exp(-x)) 

np.random.seed(22)  # try different values for the seed...

# Hyperparameters
n_hidden = 2  # number of hidden units
epochs = 500
learnrate = 0.08

n_records, n_features = features.shape
last_loss = None

# Initialize weights
W1 = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_features, n_hidden)) #W1 
W2 = np.random.normal(scale=1 / n_features ** .5,
                                        size=(n_hidden,1)) #W2
for e in range(epochs):
    del_W1 = np.zeros(W1.shape)
    del_W2 = np.zeros(W2.shape)
    for x, y in zip(features.values, targets):
        ## Forward pass ##
        h_in = np.dot(x,W1)
        h_out = sigmoid(h_in) # h = f(W1.x)
        y_in = np.dot(h_out, W2)
        y_hat = sigmoid(y_in) #y^ = f(W2.h)

        ## Backward pass ##
        # TODO: Calculate error term for the output unit
        delta0 = (y - y_hat) * sigmoid(y_in,deriv=True) # delta0 =( y-y^)f'(W2.h)
        # TODO: Calculate the error term for the hidden layer
        delta1 = np.dot(W2,delta0) * sigmoid(h_out, deriv=True) # delta1 = W2.delta0 f'(W1.x)
        
        # TODO: Update the change in weights
        del_W2 += delta0 * h_out[:, None]  #dW2 = dW2 + delta0 * h 
        del_W1 += delta1 * x[:, None]  #dW1 = dW1 + delta1 * x

    # TODO: Update weights
    W1 += learnrate * del_W1 / n_records 
    W2 += learnrate * del_W2 / n_records

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        h = sigmoid(np.dot(x, W1))
        out = sigmoid(np.dot(h,W2))
        loss = np.mean((out - targets) ** 2)

        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
hidden = sigmoid(np.dot(features_test, W1))
out = sigmoid(np.dot(hidden, W2))
predictions = out > 0.5
accuracy = np.mean(predictions == targets_test[:,None])
print("Prediction accuracy: {:.4f}".format(accuracy))
