import numpy as np


class Conv1D:

    def __call__(self, x, weight, bias, padding, dilation):
        x = np.pad(x, ((0, 0), (padding, padding)))
        _, in_len = x.shape
        out_chan, in_chan, kernel_len = weight.shape
        out_len = int(in_len - (dilation * (kernel_len-1)))

        indices = np.arange(kernel_len)[:, None] * dilation
        indices = indices + np.arange(out_len)[None]
        indices = indices.reshape(-1)

        x = x[:, indices].reshape(in_chan * kernel_len, out_len).T

        w = weight.transpose(1, 2, 0).reshape(kernel_len*in_chan, out_chan)
        y = np.dot(x, w) + bias[None]

        return y.T
