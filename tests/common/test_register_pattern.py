import torch
from xpu_graph.compiler import XpuGraph
import torch.fx as fx

import xpu_ops
xpu_ops.load_xpu_ops_npu()

def test_register_pattern():

    def _add(x, y):
        z = x + y
        return z

    def matcher(x: fx.node, y: fx.node):
        return torch.ops.aten.add.Tensor(x, y)

    def replacement(x: fx.node, y: fx.node):
        return torch.ops.aten.sub.Tensor(x, y)

    xpu_graph = XpuGraph()
    xpu_graph.get_pattern_manager().register_pattern(matcher, replacement)

    compiled = torch.compile(_add, backend=xpu_graph)
    a = torch.randn(10)
    b = torch.randn(10)
    res = compiled(a, b)

    from ..utils import is_similar
    assert(is_similar(res, a - b))