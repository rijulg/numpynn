import numpy as np

class ReLU:

    def __call__(self, x):
        return np.maximum(x, 0)