import numpy as np


class Conv1DT:

    def __call__(self, x, weight, bias, stride, padding):
        inchan, inlen = x.shape
        inchan, outchan, kernel_len = weight.shape
        outlen = int((inlen-1)*stride + kernel_len)
        x = x.T  # [inlen, inchan]
        w = weight.transpose(1, 0, 2)  # [outchan, inchan, kernel]
        temp = np.dot(x, w).transpose(0, 2, 1)  # [inlen, kernel, outchan]
        temp = temp.reshape(inlen*kernel_len, outchan)

        y = np.zeros((outlen, outchan)) + bias[None]

        out_idx = np.arange(0, outlen-2*padding)
        in_idx = (kernel_len*np.arange(inlen)[:, None] + np.arange(stride))
        in_idx = in_idx.reshape(-1)
        for i in range(kernel_len//stride):
            y[out_idx + i*stride] += temp[in_idx + i*stride]

        return y[padding:-padding].T
