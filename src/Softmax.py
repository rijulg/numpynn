import numpy as np


class Softmax:

    def __call__(self, x, axis):
        e_x = np.exp(x - np.expand_dims(np.amax(x, axis), axis))
        return e_x / np.expand_dims(np.sum(e_x, axis), axis)
