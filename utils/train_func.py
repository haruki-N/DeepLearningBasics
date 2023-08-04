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
    # to handle batch
    if pred_vec.ndim == 1:
        pred_vec = pred_vec.reshape(1, pred_vec.size)
        true_labels = true_labels.reshape(1, true_labels)
    batch_size = pred_vec.shape[0]

    # to avoid log(0)=-inf
    delta = 1e-7
    return - np.sum(true_labels * np.log(pred_vec + delta)) / batch_size


def numerical_gradient_not_batch(f, x):
    # 数値微分のための微小値h
    h = 1e-4
    grad = np.zeros_like(x)

    # 各次元について数値偏微分（中心差分）
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val

    return grad

def numerical_gradient(f, X):
    if X.ndim == 1:
        return numerical_gradient_not_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = numerical_gradient_not_batch(f, x)
        
        return grad