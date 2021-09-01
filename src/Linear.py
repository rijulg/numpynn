import numpy as np


class Linear:

    def __call__(self, x, weight, bias):
        return x @ weight.T + bias
        # return bias + np.tensordot(x, weight.T, 1)
