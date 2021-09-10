import numpy as np


class Conv1DT:

    def __call__(self, x, weight, bias, stride, padding):
        x = x.T
        in_len, in_chan = x.shape
        in_chan, out_chan, kernel_size = weight.shape
        w = weight.reshape(in_chan, -1)

        temp = np.dot(x, w).reshape(in_len, out_chan,
                                    kernel_size).transpose(0, 2, 1)
        temp = temp.reshape(in_len*kernel_size, out_chan)
        temp = np.pad(temp, ((kernel_size//2, kernel_size//2), (0, 0)))
        temp = temp.reshape(in_len+1, 2, kernel_size//2, out_chan)
        output = temp.sum(1).reshape(-1, out_chan) + bias[None]

        return output[padding:-padding].T
