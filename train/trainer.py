import numpy as np
import os
import sys
sys.path.append(os.getcwd())

from utils.train_func import numerical_gradient

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for _step in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x