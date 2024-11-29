import torch
import math


def test_convert_gelu():
    def _gelu0(x):
        return 0.5 * (1 + torch.erf(x / math.sqrt(2))) * x

    from xpu_graph.compiler import XpuGraph

    compiled_gelu0 = torch.compile(_gelu0, backend=XpuGraph())

    from xpu_graph.test_utils import is_similar

    input = torch.randn(10)
    is_similar(compiled_gelu0(input), _gelu0(input))
