import torch
import numpy as np
import torch.nn.functional as F
from ..src.Conv1DT_5 import Conv1DT as NumpyConv1DT


class Tester:
    conv1dt_numpy = NumpyConv1DT()

    def y_torch(self, x, weight, bias, stride, padding):
        x = torch.tensor(x)
        weight = torch.tensor(weight)
        bias = torch.tensor(bias)
        return F.conv_transpose1d(x, weight, bias, stride, padding).numpy()

    def y_numpy(self, x, weight, bias, stride, padding):
        return self.conv1dt_numpy(x, weight, bias, stride, padding)

    def __call__(self, inchan, outchan, kernel_len, stride, padding):
        in_len = np.random.randint(7, 64)
        x = np.random.randn(inchan, in_len)
        W = np.random.randn(inchan, outchan, kernel_len)
        B = np.random.randn(outchan)
        y1 = self.y_torch(x[None], W, B, stride, padding)[0]
        y2 = self.y_numpy(x, W, B, stride, padding)
        print(y1.shape, y2.shape)
        assert np.allclose(y1, y2)


def test():
    tester = Tester()
    for _ in range(32):
        tester(1, 1, 1, 1, 0)
        tester(1, 1, 2, 1, 0)
        tester(1, 1, 3, 1, 0)
        tester(4, 1, 3, 1, 0)
        tester(1, 2, 3, 1, 0)
        tester(1, 1, 4, 1, 0)
        tester(1, 2, 5, 1, 0)
        tester(1, 2, 7, 1, 0)

        tester(1, 1, 1, 2, 0)
        tester(1, 1, 2, 2, 0)
        tester(1, 1, 4, 3, 0)
        tester(4, 8, 4, 3, 0)
        tester(1, 1, 1, 1, 1)
        tester(1, 1, 3, 1, 1)
        tester(1, 1, 3, 2, 1)
        tester(1, 1, 3, 2, 2)

        tester(512, 256, 3, 1, 1)
        tester(256, 256, 3, 1, 1)
        tester(80, 80, 3, 1, 1)
        tester(512, 128, 13, 1, 6)
        tester(128, 128, 11, 1, 5)
        tester(128, 128, 9, 1, 4)
        tester(128, 128, 7, 1, 3)
        tester(128, 128, 5, 1, 2)
        tester(128, 128, 3, 1, 1)
        tester(128, 1, 1, 1, 0)

        tester(64, 32, 4, 2, 1)
        tester(128, 64, 4, 2, 1)
        tester(256, 128, 16, 8, 4)
        tester(512, 256, 16, 8, 4)
