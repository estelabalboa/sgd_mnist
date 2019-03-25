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
    x_valid.mean(), x_valid.std()


def show(img, title=None):
    plt.imshow(img, cmap="gray")
    if title is not None: plt.title(title)


def main():
    URL = 'http://deeplearning.net/data/mnist/'
    FILENAME = 'data/mnist/mnist.pkl.gz'
    load_mnist(FILENAME)

    # show()


if __name__ == '__main__':
    main()
