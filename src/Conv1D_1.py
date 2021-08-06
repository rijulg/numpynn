import numpy as np


class Conv1D:

    def __call__(self, x, weight, bias, padding, dilation):
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding)))
        batch_size, in_chan, in_len = x.shape
        out_chan, _, kernel_len = weight.shape
        out_len = int(in_len - (dilation * (kernel_len-1)))
        y = np.zeros((batch_size, out_chan, out_len))

        for j in range(out_chan):
            y[:, j] += bias[j]
            for i in range(out_len):
                y[:, j, i] += np.sum(weight[j] * x[:, :, i:i+kernel_len])

        return y
