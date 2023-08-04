import os
import numpy as np
from tqdm import tqdm
import sys
sys.path.append(os.getcwd())
print(sys.path)

from dataset.mnist import load_mnist
from model.simplenet import SimpleNet



def train(iters_num=1000, batch_size=32, lr=0.1):
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    train_losses = []
    train_accs = []
    test_accs = []

    train_size = x_train.shape[0]
    iter_per_epoch = max(1, train_size/batch_size)
    print('iter per epoch: ', iter_per_epoch)

    network = SimpleNet(input_dim=784, hidden_dim=64, output_dim=10)

    for iter in tqdm(range(iters_num)):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        pred = network.forward(x_batch)
        grad = network.numerical_gradient(pred, t_batch)

        for key in ('W1', 'W2', 'b1', 'b2'):
            network.params[key] -= lr * grad[key]

        loss = network.loss(pred, t_batch)
        train_losses.append(loss)

        if iter % iter_per_epoch == 0:
            train_acc = network.calc_acc(x_train, t_train)
            test_acc = network.calc_acc(x_test, t_test)
            train_accs.append(train_acc)
            test_accs.append(test_acc)

            print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

    return network, train_losses, train_accs, test_accs


def main():
    _model, train_losses, train_accs, test_accs = train(iters_num=100)
    print("Train Ends: train acc, test acc | " + str(train_accs[-1]) + ", " + str(test_accs[-1]))

if __name__ == '__main__':
    main()
