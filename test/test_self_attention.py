import torch
import numpy as np
import torch.nn.functional as F
from ..src.SelfAttention import SelfAttention as NumpySelfAttention


class Tester:
    self_attention_numpy = NumpySelfAttention()

    @torch.no_grad()
    def process(self, in_len, embed_dim, num_heads):
        attn = torch.nn.MultiheadAttention(embed_dim, num_heads)
        src = np.random.randn(in_len, embed_dim).astype('float32')
        t_src = torch.tensor(src)[:, None]
        expected_output = attn(t_src, t_src, t_src)[0][:, 0]
        return (
            t_src[:, 0].numpy(),
            attn.in_proj_weight.numpy(),
            attn.in_proj_bias.numpy(),
            attn.out_proj.weight.numpy(),
            attn.out_proj.bias.numpy(),
            expected_output.numpy()
        )

    def __call__(self, num_heads):
        in_len = np.random.randint(1, 128)
        embed_dim = np.random.randint(1, 256)
        embed_dim = 256
        (
            src,
            in_kernel,
            in_bias,
            out_kernel,
            out_bias,
            expected_output
        ) = self.process(in_len, embed_dim, num_heads)

        output = self.self_attention_numpy(
            num_heads, src, in_kernel.T, in_bias, out_kernel.T, out_bias
        )

        assert expected_output.shape == output.shape

        err = np.sum(np.abs(expected_output - output))

        assert err < 1e-3


def test_one_head():
    tester = Tester()
    for _ in range(32):
        tester(1)


def test_two_head():
    tester = Tester()
    for _ in range(32):
        tester(2)
