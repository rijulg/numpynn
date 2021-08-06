import numpy as np


class Conv1D:

    def __call__(self, x, weight, bias, padding, dilation):
        x = np.pad(x, ((0, 0), (padding, padding))).T
        in_len, _ = x.shape
        out_chan, _, kernel_len = weight.shape
        out_len = int(in_len - (dilation * (kernel_len-1)))
        y = np.zeros((out_chan, out_len))
        weight = weight.transpose(2, 1, 0)

        y += bias[:, None]
        for i in range(out_len):
            y[:, i] += np.tensordot(x[i:i+kernel_len], weight)

        return y
