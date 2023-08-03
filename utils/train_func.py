import numpy as np



def relu(x):
    return np.maximum(0, x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(vec_array):
    # to avoid overflow
    vec_array -= np.max(vec_array)

    exp_vecs = np.exp(vec_array)
    sum_exp_vecs = np.sum(exp_vecs)

    return exp_vecs / sum_exp_vecs


def mean_squared_error(pred_vec, true_labels):
    return 0.5 * np.sum((pred_vec - true_labels) ** 2)


def cross_entropy_loss(pred_vec, true_labels):
    # to avoid log(0)=-inf
    delta = 1e-7
    return - np.sum(true_labels * np.log(pred_vec + delta))

