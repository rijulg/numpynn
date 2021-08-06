import torch
import numpy as np
import torch.nn.functional as F
from ..src.LayerNorm import LayerNorm as NumpyLayerNorm


class Tester:
    layernorm_numpy = NumpyLayerNorm()

    def y_torch(self, x, shape, weight, bias, eps):
        x = torch.tensor(x)
        weight = torch.tensor(weight)
        bias = torch.tensor(bias)
        return F.layer_norm(x, shape, weight, bias, eps).numpy()

    def y_numpy(self, x, shape, weight, bias, eps):
        return self.layernorm_numpy(x, shape, weight, bias, eps)

    def __call__(self):
        shape = np.random.randint(1, 128)
        seqlen = np.random.randint(1, 128)
        x = np.random.randn(seqlen, shape)
        W = np.random.randn(shape)
        B = np.random.randn(shape)
        eps = 1e-5
        y1 = self.y_torch(x, (shape,), W, B, eps)
        y2 = self.y_numpy(x, (shape,), W, B, eps)
        assert np.allclose(y1, y2)


def test():
    tester = Tester()
    for _ in range(32):
        tester()
