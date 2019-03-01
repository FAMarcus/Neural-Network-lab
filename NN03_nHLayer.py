#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 10:44 2019
@author: F.A.Marcus
Code improvement from original: Udacity - nd109 Neural Network
"""
import numpy as np
import time

from functions_lib import sigmoid, hyp_tan
from graphics_lib import plot_cplx, plot_accuracy
from data_UCLA import features, targets, features_test, targets_test

np.random.seed(42)

# Hyperparameters
hidden = [3,2]   # hidden layers.
epochs = 800
learnrate = 0.1

n_records, n_features = features.shape
n_inputs, n_rows = np.matrix(targets).shape
#obs: dados de entrada e saida devem ser conhecidos, não é correto deixar para o programa determinar tais valores.

# Dados como arrays do numpy
features = np.array(features.values); targets = np.array(targets)
features_test = np.array(features_test.values) 
targets_test  = np.array(targets_test)

errors = []; errors_test = []; acc = []
training_loss = None; last_loss = None

print("Hidden Layers: (",str(hidden),")","; Learning rate: ",learnrate)

# montando uma lista com o numero de nós de cada camada
layers = [n_features]
layers.extend([h for h in hidden])
layers.append(n_inputs)
n_layers = len(layers)

# Initialize weights and biases
W = [np.random.normal(scale=1.0 / n_features ** 0.5, size=(i,j)) 
      for i,j in zip( layers[:-1], layers[1:]) ]

t0 = time.clock()
for e in range(epochs):
    del_W = [ np.zeros(w.shape) for w in W] 
    for x, y in zip(features, targets):
        ## Forward pass ##
        z = None; zL = []
        a = x ; aL = [x]
        for w in W[:-1]:
            z = np.dot(a,w)
            zL.append(z)
            a = hyp_tan(z)
            aL.append(a)
        z = np.dot(a,W[-1])
        zL.append(z)
        a = sigmoid(z)
        aL.append(a)
        
        ## Backward pass ##
        delta = (y - aL[-1]) * sigmoid(zL[-1], deriv=True)
        del_W[-1] += np.outer(aL[-2], delta)
        for l in range(2,n_layers):
            delta = np.dot(W[-l+1],delta) * hyp_tan(zL[-l], deriv=True)
            del_W[-l] += np.outer(aL[-l-1], delta)

    # TODO: Update weights and biases
    W = [w + (learnrate * dw / n_records) for w, dw in zip(W, del_W) ]
    
    # compute the loss
    a_train = features
    for w in W[:-1]:
        a_train = hyp_tan(np.dot(a_train,w))
    a_train = sigmoid(np.dot(a_train,W[-1]))     
    loss = 0.5 * np.mean((a_train - targets[:, None])**2)
    errors.append(loss)
    
    ## Validation loss
    a_test = features_test
    for w in W[:-1]:
        a_test = hyp_tan(np.dot(a_test,w))
    a_test = sigmoid(np.dot(a_test,W[-1]))    
    testing_loss = 0.5 * np.mean((a_test - targets_test[:, None])**2)
    errors_test.append(testing_loss)
    
    predictions = a_test > 0.5
    accuracy = np.mean(predictions == targets_test[:, None])
    acc.append(accuracy)
    
    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:
        if last_loss and last_loss < loss:
            print("Epoch:", e, "; Accuracy: {:.3f} ".format(accuracy), "; Train loss: {:.5f}".format(loss), " - Loss Increasing; ")
        else:
            print("Epoch:", e, "; Accuracy: {:.3f} ".format(accuracy), "; Train loss: {:.5f}".format(loss))
        last_loss = loss

tf = time.clock()

# Results Summary
print("#############################################")
print("Training time: {:.2f}".format(tf - t0), " s")
print("Prediction accuracy: {:.4f}".format(accuracy))
print("Major accuracy value: {:.3f}".format(np.max(acc)))
print("Epoch of max accuracy value:", np.argmax(acc))
print("Min error value: {:.4f}".format(np.min(errors)))
print("Epoch of min error value:", np.argmin(errors))

# Plot error
plot_cplx(errors,errors_test,hidden,accuracy,learnrate)
plot_accuracy(acc)
###########################################################################
## On screen
#Hidden Layers: ( [3, 2] ) ; Learning rate:  0.07
#Epoch: 0 ; Accuracy: 0.250  ; Train loss: 0.16526
#Epoch: 50 ; Accuracy: 0.250  ; Train loss: 0.14260
#Epoch: 100 ; Accuracy: 0.250  ; Train loss: 0.13196
#Epoch: 150 ; Accuracy: 0.250  ; Train loss: 0.12789
#Epoch: 200 ; Accuracy: 0.250  ; Train loss: 0.12625
#Epoch: 250 ; Accuracy: 0.250  ; Train loss: 0.12555
#Epoch: 300 ; Accuracy: 0.275  ; Train loss: 0.12525
#Epoch: 350 ; Accuracy: 0.375  ; Train loss: 0.12511
#Epoch: 400 ; Accuracy: 0.600  ; Train loss: 0.12505
#Epoch: 450 ; Accuracy: 0.550  ; Train loss: 0.12502
##############################################
#Training time: 23.88  s
#Prediction accuracy: 0.525
#Major accuracy value: 0.600
#Epoch of max accuracy value: 397
#Min error value: 0.1250
#Epoch of min error value: 499
#Min error distance: 5.34e-05
#Epoch of min error distance: 499