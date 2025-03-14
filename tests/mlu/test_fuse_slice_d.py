import pytest

import torch
import torch_mlu
import torch.nn.functional as F
import xpu_graph
from xpu_graph import OptLevel
from xpu_graph.test_utils import is_similar

device = "mlu:0"
aten = torch.ops.aten


def fn0(a, b, c, d):
    c = torch.cat([a[:, 16:32], b[:, 16:32], c[:, 16:32], d], 1)
    return c

def slice_test(xpu_graph_backend, func):
    batch = 2048
    start_w = 300
    input_list = []
    slice_list = []
    a = torch.randn(batch, start_w).to(device=device).to(torch.float32)
    b = torch.randn(batch, start_w).to(device=device).to(torch.float32)
    c = torch.randn(batch, start_w).to(device=device).to(torch.float32)
    d = torch.randn(batch, 32).to(device=device).to(torch.float32)
    ref = torch.randn(batch, 3 * 16).to(device=device).to(torch.float32)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=None)
    norm1 = func(a, b, c, d)
    for i in range(10):
        norm0 = compiled(a, b, c, d)
    assert torch.equal(norm0, norm1)
    '''

    a0, b0, c0 = (
        a.clone().requires_grad_(),
        b.clone().requires_grad_(),
        c.clone().requires_grad_(),
    )
    norm0 = compiled(a0,b0,c0)
    loss0 = F.mse_loss(norm0, ref)
    loss0.backward()

    a1, b1, c1 = (
        a.clone().requires_grad_(),
        b.clone().requires_grad_(),
        c.clone().requires_grad_(),
    )
    norm1 = func(a1,b1,c1)
    loss1 = F.mse_loss(norm1, ref)
    loss1.backward()

    assert is_similar(norm0.detach(), norm1.detach())
    assert is_similar(loss0.detach(), loss1.detach())

    assert is_similar(a0.grad, a1.grad)
    assert is_similar(b0.grad, b1.grad)
    assert is_similar(c0.grad, c1.grad)
    '''


class TestSliced:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False, freeze=True, opt_level=OptLevel.level2
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
        ],
    )
    def test_slice_patterns(self, pattern_func):
        slice_test(self.xpu_graph_backend, pattern_func)


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False, opt_level=OptLevel.level1, vendor_compiler_config=None, debug=True
    )
    slice_test(xpu_graph_backend, fn0)
