import torch
import numpy as np
import torch.nn.functional as F
from ..src.Softmax import Softmax as NumpySoftmax


class Tester:
    softmax_numpy = NumpySoftmax()

    def y_torch(self, x, axis):
        return F.softmax(torch.tensor(x), axis).numpy()

    def y_numpy(self, x, axis):
        return self.softmax_numpy(x, axis)

    def __call__(self):
        max_dim_len = 32
        dim = np.random.randint(1, 4)
        axis = np.random.randint(0, dim)

        dim_lens = [np.random.randint(1, max_dim_len) for _ in range(dim)]
        src = np.random.randn(*dim_lens).astype('float32')

        err = np.sum(np.abs(self.y_torch(src, axis) - self.y_numpy(src, axis)))
        assert err < 1e-4


def test():
    tester = Tester()
    for _ in range(32):
        tester()
