import torch
import numpy as np
import torch.nn.functional as F
from ..src.Linear import Linear as NumpyLinear


class Tester:
    linear_numpy = NumpyLinear()

    def y_torch(self, x, weight, bias):
        x = torch.tensor(x)
        weight = torch.tensor(weight)
        bias = torch.tensor(bias)
        return F.linear(x, weight, bias).numpy()

    def y_numpy(self, x, weight, bias):
        return self.linear_numpy(x, weight, bias)

    def __call__(self):
        x = np.random.randn(32, 128)
        W = np.random.randn(64, 128)
        B = np.random.randn(64)
        assert np.array_equal(
            self.y_torch(x, W, B),
            self.y_numpy(x, W, B)
        )


def test():
    tester = Tester()
    for _ in range(32):
        tester()
