import numpy as np


class Conv1DT:

    def __call__(self, x, weight, bias, stride, padding):
        inchan, inlen = x.shape
        inchan, outchan, kernel_len = weight.shape
        outlen = int((inlen-1)*stride - 2*padding + kernel_len)
        x = x.T  # [inlen, inchan]
        w = weight.transpose(1, 0, 2)  # [outchan, inchan, kernel]
        temp = np.dot(x, w)  # [inlen, outchan, kernel]
        y = np.zeros((outchan, outlen + 2*padding))
        for i in range(inlen):
            yi = stride*i
            y[:, yi:yi+kernel_len] += temp[i]
        y += bias[:, None]
        if padding > 0:
            return y[:, padding:-padding]
        return y
