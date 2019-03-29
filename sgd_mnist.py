from IPython.core.display import display
from fastai.imports import *
import torch
import torch.nn as nn
from fastai.metrics import *
from fastai.model import *
from fastai.dataset import *
from fastai.old.fastai.core import to_np, V


import os
# path = 'data/mnist/'
# os.makedirs(path, exist_ok=True)



def load_mnist(filename):
    """

    :param filename:
    :return:
    """
    return pickle.load(gzip.open(filename, 'rb'), encoding='latin-1')

    # get_data(URL + FILENAME, path + FILENAME)
    get_data(filename)
    ((x, y), (x_valid, y_valid), _) = load_mnist(path + FILENAME)

    display(type(x), x.shape, type(y), y.shape)

    mean = x.mean()
    std = x.std()

    x = (x - mean) / std
    mean, std, x.mean(), x.std()

    x_valid = (x_valid - mean) / std
    return x_valid.mean()


def show(img, title=None):
    """

    :param img:
    :param title:
    :return:
    """
    plt.imshow(img, cmap="gray")
    if title is not None: plt.title(title)


def plots(ims, figsize=(12, 6), rows=2, titles=None):
    """

    :param ims:
    :param figsize:
    :param rows:
    :param titles:
    :return:
    """
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], cmap='gray')


def binary_loss(y, p):
    """
    LOSS FUNCTIONS AND METRICS
    :param y:
    :param p:
    :return:
    """
    return np.mean(-(y * np.log(p) + (1-y)*np.log(1-p)))


def get_weights(*dims):
    """

    :param dims:
    :return:
    """
    return nn.Parameter(torch.randn(dims)/dims[0])


def softmax(x):
    """

    :param x:
    :return:
    """
    return torch.exp(x)/(torch.exp(x).sum(dim=1)[:, None])


class LogReg(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1_w = get_weights(28*28, 10)  # Layer 1 weights
        self.l1_b = get_weights(10)         # Layer 1 bias

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = x @ self.l1_w + self.l1_b
        return torch.log(softmax(x))


def score(x, y):
    net2 = LogReg().cuda()
    opt = optim.Adam(net2.parameters())
    y_pred = to_np(net2(V(x)))
    return np.sum(y_pred.argmax(axis=1) == to_np(y))/len(y_pred)


def main():
    url = 'http://deeplearning.net/data/mnist/'
    filename = 'data/mnist/mnist.pkl.gz'
    x_valid = load_mnist(filename)

    display(x_valid)


if __name__ == '__main__':
    main()
