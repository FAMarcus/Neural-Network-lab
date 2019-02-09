import numpy as np

### Activation functions
## Sigmoid 
def sigmoid(x,deriv=False):
    """ Calculate sigmoid and its derivative."""
    if (deriv==True):
       return x*(1.0-x)
    return 1/(1+np.exp(-x)) 

## Hyperbolic tangent 
def hyp_tan(x,deriv=False):
    """ Calculate tanh and its derivative."""
    if (deriv==True):
       return 1 - np.tanh(x)**2
    return np.tanh(x)     

### Loss functions
## Binary cross-entropy error
def log_error(y,y_hat):
    return np.mean(-y*np.log(y_hat) - (1-y)*np.log(1-y_hat))

## Least mean square error
def sqr_error(y,y_hat):
    return np.mean((y-y_hat)**2)

def cost_sqr(y,y_hat):
    return (y-y_hat)
          
    
    
