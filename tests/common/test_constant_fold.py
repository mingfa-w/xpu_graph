import torch
from xpu_graph.compiler import XpuGraph


def test_constant_folding():
    class Foo(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.rand(128, 128), False)

        @torch.no_grad()
        def forward(self, x):
            weight = torch.relu(torch.relu(self.weight))
            return torch.matmul(x, weight)

    foo = Foo()
    xpu_graph = XpuGraph()
    compiled_foo = torch.compile(foo, backend=xpu_graph, dynamic=False)
    res = compiled_foo(torch.ones(128, 128))
    expect = foo(torch.ones(128, 128))

    from xpu_graph.test_utils import is_similar

    is_similar(res, expect)
