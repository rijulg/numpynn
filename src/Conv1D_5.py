import numpy as np


class Conv1D:

    def __call__(self, x, weight, bias):
        W = weight.transpose(1, 0, 2)[:, :, 0]
        y = bias + np.tensordot(x.T, W, 1)
        return y.T
