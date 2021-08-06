import numpy as np


class LayerNorm:

    def __call__(self, x, shape, weight, bias, eps):
        mu = np.mean(x, axis=-1)[:, None]
        sig = np.var(x, axis=-1)[:, None]
        y = (x - mu) / np.sqrt(sig + eps)
        y = weight * y + bias
        return y
