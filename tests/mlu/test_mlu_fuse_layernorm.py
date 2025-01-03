import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_mlu
from xpu_graph.config import OptLevel
import torch_mlu_ops
import xpu_graph
from xpu_graph.test_utils import is_similar


device = "mlu:0"
data_type = torch.bfloat16
aten = torch.ops.aten


def fn0(inputs, residual, weight, bias):
    inputs_ = inputs + residual
    output = torch.layer_norm(
        inputs_, normalized_shape=[1024], weight=weight, bias=bias, eps=1e-5
    )
    return output


def layernorm_test(xpu_graph, func):
    if func == fn0:
        inputs = torch.randn((8, 1024), device=device, dtype=data_type)
        residual = torch.randn((8, 1024), device=device, dtype=data_type)
        weight = torch.randn((1024), device=device, dtype=data_type)
        bias = None
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    res = compiled(inputs, residual, weight, bias)
    res1 = func(inputs, residual, weight, bias)
    assert is_similar(res1, res)


class TestLayerNorm:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig()
        config.target = xpu_graph.config.Target.mlu
        config.vendor_compiler = {"mode": "reduce-overhead"}
        config.opt_level = OptLevel.level2
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0],
    )
    def test_slice_patterns(self, pattern_func):
        layernorm_test(self.xpu_graph, pattern_func)


if __name__ == "__main__":
    pytest.main()
