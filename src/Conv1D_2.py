import numpy as np


class Conv1D:

    def __call__(self, x, weight, bias, padding, dilation):
        x = np.pad(x, ((0, 0), (0, 0), (padding, padding)))
        batch_size, _, in_len = x.shape
        out_chan, _, kernel_len = weight.shape
        out_len = int(in_len - (dilation * (kernel_len-1)))
        y = np.zeros((batch_size, out_chan, out_len))
        weight = weight.transpose(2, 1, 0)

        y += bias[None, :, None]
        for i in range(out_len):
            _x = x[:, :, i:i+kernel_len].transpose(0, 2, 1)
            y[:, :, i] += np.tensordot(_x, weight)

        return y
