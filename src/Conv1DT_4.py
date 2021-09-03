import numpy as np


class Conv1DT:

    def __call__(self, x, weight, bias, stride, padding):
        x = x.T
        w = weight.transpose(2, 0, 1)
        kernel_len, _, outchan = w.shape

        temp = np.dot(x, w)  # [inlen, kernel, outchan]
        temp = temp.reshape(-1, outchan)
        temp = np.pad(temp, ((kernel_len//2, kernel_len//2), (0, 0)))
        temp = temp.reshape(-1, 2, kernel_len//2, outchan)

        output = temp.sum(1).reshape(-1, outchan) + bias[None]

        return output[padding:-padding].T
