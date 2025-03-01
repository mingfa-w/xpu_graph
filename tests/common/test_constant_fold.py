import torch
from xpu_graph.compiler import XpuGraph


def test_constant_folding():
    class Foo(torch.nn.Module):
        def __init__(self):
            # 调用父类的初始化方法
            super().__init__()
            # 初始化权重参数，使用随机数初始化，大小为128x128
            self.weight = torch.nn.Parameter(torch.rand(128, 128), False)

        @torch.no_grad()
        def forward(self, x):
            return torch.add(x, self.weight)

    foo = Foo()
    from xpu_graph.config import XpuGraphConfig
    xconf = XpuGraphConfig(constant_folding=True)
    xpu_graph = XpuGraph(config=xconf)
    compiled_foo = torch.compile(foo, backend=xpu_graph, dynamic=False)
    res = compiled_foo(torch.ones(128, 128))
    expect = foo(torch.ones(128, 128))

    from xpu_graph.test_utils import is_similar

    is_similar(res, expect)