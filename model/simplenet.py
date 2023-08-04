import numpy as np
import os
import sys
sys.path.append(os.getcwd())
from utils.train_func import *


# simple two layers network
class SimpleNet:
    def __init__(self, input_dim, hidden_dim, output_dim, weight_init_std=0.01):
        # G分布で初期化
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)

        self.params['W2'] = weight_init_std * np.random.randn(hidden_dim, output_dim)
        self.params['b2'] = np.zeros(output_dim)

    @staticmethod
    def active_function(x):
        return sigmoid(x)


    def forward(self, x):
        x = np.dot(x, self.params['W1']) + self.params['b1']
        x = sigmoid(x)
        x = np.dot(x, self.params['W2']) + self.params['b2']
        y = softmax(x)

        return y


    def loss(self, pred, true):
        return cross_entropy_loss(pred, true)


    def calc_acc(self, pred, true):
        y = np.argmax(pred, axis=1)
        t = np.argmax(true, axis=1)

        acc = np.sum(y == t) / float(pred.shape[0])
        return acc
    

    def numerical_gradient(self, pred, true):   # backprop. は使わず, 数値微分で最適化
        loss_W = lambda W: self.loss(pred, true)

        grads = {}

        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads
