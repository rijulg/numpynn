import math
import numpy as np


class SelfAttention:

    def linear(self, x, kernel, bias):
        return bias + np.tensordot(x, kernel, 1)

    def softmax(self, x, axis):
        e_x = np.exp(x - np.expand_dims(np.amax(x, axis), axis))
        return e_x / np.expand_dims(np.sum(e_x, axis), axis)

    def __call__(self, num_heads, x, in_kernel, in_bias, out_kernel, out_bias):
        if num_heads == 1:
            return self.one_head(x, in_kernel, in_bias, out_kernel, out_bias)
        if num_heads == 2:
            return self.two_head(x, in_kernel, in_bias, out_kernel, out_bias)

    def one_head(self, x, in_kernel, in_bias, out_kernel, out_bias):
        embed_dim = out_bias.shape[0]
        eps = math.sqrt(embed_dim)
        x = self.linear(x, in_kernel, in_bias)
        q = x[:, embed_dim*0:embed_dim*1]
        k = x[:, embed_dim*1:embed_dim*2]
        v = x[:, embed_dim*2:embed_dim*3]

        attn = np.dot(q, k.T) / eps
        attn = self.softmax(attn, 1)
        output = np.dot(attn, v)

        return self.linear(output, out_kernel, out_bias)

    def two_head(self, x, in_kernel, in_bias, out_kernel, out_bias):
        embed_dim = out_bias.shape[0]
        head_dim = embed_dim // 2
        eps = math.sqrt(head_dim)

        x = self.linear(x, in_kernel, in_bias)

        q0 = x[:, head_dim*0:head_dim*1]
        q1 = x[:, head_dim*1:head_dim*2]
        k0 = x[:, head_dim*2:head_dim*3]
        k1 = x[:, head_dim*3:head_dim*4]
        v0 = x[:, head_dim*4:head_dim*5]
        v1 = x[:, head_dim*5:head_dim*6]

        o0 = np.dot(self.softmax(np.dot(q0, k0.T) / eps, 1), v0)
        o1 = np.dot(self.softmax(np.dot(q1, k1.T) / eps, 1), v1)

        output = np.concatenate((o0, o1), axis=1)

        return self.linear(output, out_kernel, out_bias)
