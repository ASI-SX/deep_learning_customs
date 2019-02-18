import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x)  # オーバーフロー対策
    return np.exp(x) / np.sum(np.exp(x))


def onehot_encoding(x, classes):
    output = np.identity(classes)[x]
    return output


def onehot_decoding(x):
    x = np.asarray(x)
    y = np.argmax(x, axis=-1)
    return y
