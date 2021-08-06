import torch
import numpy as np
import torch.nn.functional as F
from ..src.Conv1D_2 import Conv1D as NumpyConv1D


class Tester:
    conv1d_numpy = NumpyConv1D()

    def y_torch(self, x, weight, bias, padding, dilation):
        x = torch.tensor(x)
        weight = torch.tensor(weight)
        bias = torch.tensor(bias)
        stride = 1
        return F.conv1d(x, weight, bias, stride, padding, dilation).numpy()

    def y_numpy(self, x, weight, bias, padding, dilation):
        return self.conv1d_numpy(x, weight, bias, padding, dilation)

    def __call__(self):
        batch_size = np.random.randint(1, 32)
        in_chan = np.random.randint(1, 32)
        out_chan = np.random.randint(1, 32)
        in_len = np.random.randint(3, 32)
        kernel_len = np.random.randint(1, in_len)
        padding = np.random.randint(0, 32)
        dilation = 1

        x = np.random.randn(batch_size, in_chan, in_len)
        W = np.random.randn(out_chan, in_chan, kernel_len)
        B = np.random.randn(out_chan)
        y1 = self.y_torch(x, W, B, padding, dilation)
        y2 = self.y_numpy(x, W, B, padding, dilation)

        assert np.allclose(y1, y2)


def test():
    tester = Tester()
    for _ in range(32):
        tester()
