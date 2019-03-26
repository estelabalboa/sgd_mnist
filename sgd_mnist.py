from IPython.core.display import display
from fastai.imports import *
import torch
import os

# path = 'data/mnist/'
# os.makedirs(path, exist_ok=True)


def load_mnist(filename):
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
    plt.imshow(img, cmap="gray")
    if title is not None: plt.title(title)


def plots(ims, figsize=(12, 6), rows=2, titles=None):
    f = plt.figure(figsize=figsize)
    cols = len(ims)//rows
    for i in range(len(ims)):
        sp = f.add_subplot(rows, cols, i+1)
        sp.axis('Off')
        if titles is not None: sp.set_title(titles[i], fontsize=16)
        plt.imshow(ims[i], cmap='gray')


def main():
    url = 'http://deeplearning.net/data/mnist/'
    filename = 'data/mnist/mnist.pkl.gz'
    x_valid = load_mnist(filename)

    display(x_valid)


if __name__ == '__main__':
    main()
