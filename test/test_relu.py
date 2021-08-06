import torch
import numpy as np
from ..src.ReLU import ReLU as NumpyReLU


class Tester:

    relu_numpy = NumpyReLU()

    def y_torch(self, x):
        return torch.relu(torch.tensor(x)).numpy()

    def y_numpy(self, x):
        return self.relu_numpy(x)

    def __call__(self):
        x = np.random.randn(128)
        assert np.array_equal(
            self.y_torch(x),
            self.y_numpy(x)
        )


def test():
    tester = Tester()
    for _ in range(32):
        tester()
