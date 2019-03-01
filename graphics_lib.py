#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 22:07:14 2018

@author: alberto
"""
import matplotlib.pyplot as plt
import numpy as np

# Plot Model Complex Graph
def plot_cplx(errors,errors_test,hl,accuracy=None,learnrate=None):
    plt.title("Model Complexity Graph \n" + "h_layers: (" + str(hl) + ")" \
              + ", l_rate: " + str(learnrate) + ", Acc: {:.3f}".format(accuracy))
    plt.xlabel("Number of Epochs")
    plt.ylabel("Error")
    plt.plot(errors, label='Training')
    plt.plot(errors_test, label='Validation')
    plt.legend()
    plt.grid(axis='both')
    plt.show()

def plot_accuracy(acc):
    plt.title("Accuracy Plot")
    plt.xlabel('Number of epochs')
    plt.ylabel('Accuracy')
    plt.grid(axis='both')
    plt.plot(acc)
    plt.show()
