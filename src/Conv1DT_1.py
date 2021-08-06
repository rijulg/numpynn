import numpy as np


class Conv1DT:

    def __call__(self, x, weight, bias, stride, padding):
        inchan, inlen = x.shape
        inchan, outchan, kernel_len = weight.shape
        outlen = int((inlen-1)*stride - 2*padding + kernel_len)
        x = x.T
        y = np.zeros((outchan, outlen))
        for i in range(inlen):
            for k in range(kernel_len):
                _w = weight[:, :, k]
                _x = x[i, :]
                y_i = i*stride + k - padding
                if y_i > -1 and y_i < outlen:
                    y[:, y_i] += _x @ _w
        y += bias[:, None]
        return y
