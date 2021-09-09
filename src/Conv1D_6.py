import numpy as np


class Conv1D:

    def __call__(self, x, weight, bias, padding, dilation):
        x = np.pad(x, ((0, 0), (padding, padding))).T
        in_len, _ = x.shape
        out_chan, in_chan, kernel_len = weight.shape
        out_len = int(in_len - (dilation * (kernel_len-1)))

        k_indices = np.arange(kernel_len)
        indices = np.tile(k_indices[None], (out_len, 1))
        indices += np.arange(out_len)[:, None] + (k_indices*(dilation-1))

        x = x[indices].transpose(0, 2, 1).reshape(out_len, kernel_len*in_chan)
        w = weight.transpose(1, 2, 0).reshape(kernel_len*in_chan, out_chan)
        y = np.dot(x, w) + bias[None]

        return y.T
