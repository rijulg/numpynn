import numpy as np


class Conv1D:

    def __call__(self, x, weight, bias, padding, dilation):
        x = np.pad(x, ((0, 0), (padding, padding))).T
        in_len, _ = x.shape
        _, _, kernel_len = weight.shape
        out_len = int(in_len - (dilation * (kernel_len-1)))

        k_indices = np.arange(kernel_len)
        indices = np.tile(k_indices[None], (out_len, 1))
        indices += np.arange(out_len)[:, None] + (k_indices*(dilation-1))

        # out_len, kernel_len, in_chan
        x = x[indices]
        # kernel_len, in_chan, out_chan
        w = weight.transpose(2, 1, 0)
        # out_len, out_chan
        y = np.tensordot(x, w, ((1, 2), (0, 1))) + bias[None]

        return y.T
