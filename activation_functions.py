#!/usr/bin/python

"""
Created on July 11, 2018

@author: flg-ma
@attention: compare different activation functions for ML
@contact: albus.marcel@gmail.com (Marcel Albus)
@version: 1.0.0

#############################################################################################

History:
- v1.0.0: first init
"""

import numpy as np
from matplotlib import pyplot as plt


class ActivationFunctions():
    """ Class with different activation functions for NN implemented """
    def __init__(self):
        self.functions = ['ReLU', 'Softsign', 'PReLU', 'Sigmoid', 'ELU']

    def ReLU(self, data):
        """ 
        $f(x) = \begin{cases}
                0 & \text{f"ur } x < 0 \\
                x & \text{f"ur } x >= 0
                \end{cases}$
        returns the given data as ReLU (Rectified linear unit) function
        data: numpy array
        return: numpy array as ReLU function
        """
        data[data < 0] = 0
        return data

    def softsign(self, data):
        """ 
        $f(x) = \frac{x}{1 + \abs{x}}$
        returns the given data as softsign function
        data: numpy array
        return: numpy array as softsign function 
        """
        return (data / (1 + np.abs(data)))

    def PReLU(self, data, alpha):
        """ 
        $f(x) = \begin{cases}
                \alpha \cdot 0 & \text{f"ur } x < 0 \\
                x & \text{f"ur } x >= 0
                \end{cases}$
        returns the given data as PReLU (parametric rectified linear unit) function
        data: numpy array
        return: numpy array as PReLU function 
        """
        data[ data >= 0] = alpha * data[ data >= 0]
        return data
            
    def sigmoid(self, data):
        """ 
        $f(x) = \frac{1}{1 + exp{-x}}$
        returns the given data as sigmoid (logistic) function
        data: numpy array
        return: numpy array as sigmoid function 
        """
        return (1 / (1 + np.exp(- data)))

    def ELU(self, data, alpha):
        """ 
        $f(x) = \begin{cases}
                \alpha (\exp{x} - 1) & \text{f"ur } x < 0 \\
                x & \text{f"ur } x >= 0
                \end{cases}$
        returns the given data as ELU (exponential linear unit) function
        data: numpy array
        return: numpy array as ELU function 
        """
        data[ data < 0] = alpha * (np.exp(data[ data < 0]) - 1)
        return data


X_MIN = -5
X_MAX = 5
Y_MIN = -5
Y_MAX = 5
A = np.linspace(X_MIN, X_MAX, 1000)
x = [a for a in np.linspace(X_MIN, X_MAX, len(A))]

if __name__ == '__main__':
    af = ActivationFunctions()

    ###########
    # plotting
    ###########
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle('Activation Functions', fontsize=15, fontweight='normal')
    ax[0, 0].plot(x, A.copy(), 'b')
    ax[0, 0].legend(['Normal'])
    ax[0, 0].axis([X_MIN, X_MAX, Y_MIN, Y_MAX])
    ax[0, 0].grid()

    ax[0, 1].plot(x, af.softsign(A.copy()), 'b')
    ax[0, 1].legend(['Softsign'])
    ax[0, 1].axis([X_MIN, X_MAX, Y_MIN, Y_MAX])
    ax[0, 1].grid()

    alpha = 2.0
    ax[0, 2].plot(x, af.PReLU(A.copy(), alpha), 'b')
    ax[0, 2].legend(['PReLU\nAlpha: ' + str(alpha)])
    ax[0, 2].axis([X_MIN, X_MAX, Y_MIN, Y_MAX])
    ax[0, 2].grid()

    ax[1, 0].plot(x, af.ReLU(A.copy()), 'r')
    ax[1, 0].legend(['ReLU'])
    ax[1, 0].axis([X_MIN, X_MAX, Y_MIN, Y_MAX])
    ax[1, 0].grid()

    ax[1, 1].plot(x, af.sigmoid(A.copy()), 'r')
    ax[1, 1].legend(['Sigmoid'])
    ax[1, 1].axis([X_MIN, X_MAX, Y_MIN, Y_MAX])
    ax[1, 1].grid()

    beta = .7
    ax[1, 2].plot(x, af.ELU(A.copy(), beta), 'r')
    ax[1, 2].legend(['ELU\nAlpha: ' + str(beta)])
    ax[1, 2].axis([X_MIN, X_MAX, Y_MIN, Y_MAX])
    ax[1, 2].grid()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
