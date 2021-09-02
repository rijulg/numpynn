import numpy as np


class Conv1DT:

    def __call__(self, x, weight, bias, stride, padding):
        x = x.T
        w = weight.transpose(2, 0, 1)
        inlen, inchan = x.shape
        kernel_len, inchan, outchan = w.shape

        temp = np.dot(x, w)  # [inlen, kernel, outchan]

        temp = temp.reshape(inlen*kernel_len, outchan)

        outlen = int((inlen-1)*stride + kernel_len)
        y = np.zeros((outlen, outchan)) + bias[None]

        out_idx = np.arange(0, outlen-2*padding)
        in_idx = (kernel_len*np.arange(inlen)[:, None] + np.arange(stride))
        in_idx = in_idx.reshape(-1)
        for i in range(kernel_len//stride):
            y[out_idx + i*stride] += temp[in_idx + i*stride]

        return y[padding:-padding].T
