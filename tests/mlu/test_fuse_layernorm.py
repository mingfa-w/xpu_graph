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
data_type = torch.float16
aten = torch.ops.aten


def fn0(inputs, residual, weight, bias):
    inputs_ = inputs + residual
    output = torch.layer_norm(
        inputs_, normalized_shape=[1024], weight=weight, bias=bias, eps=1e-5
    )
    return output


def fn1(inputs, residual, weight, bias):
    inputs_ = inputs + residual
    output = torch.layer_norm(
        inputs_, normalized_shape=[1024], weight=weight, bias=bias, eps=1e-5
    )
    return output, inputs_


def layernorm_test(xpu_graph, func):
    inputs = torch.randn((8, 1024), device=device, dtype=data_type)
    residual = torch.randn((8, 1024), device=device, dtype=data_type)
    weight = torch.randn((1024), device=device, dtype=data_type)
    bias = None
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    if func == fn0:
        norm = compiled(inputs, residual, weight, bias)
        norm1 = func(inputs, residual, weight, bias)
        assert is_similar(norm1, norm)
    if func == fn1:
        norm, res = compiled(inputs, residual, weight, bias)
        norm1, res1 = func(inputs, residual, weight, bias)
        assert is_similar(norm1, norm)
        assert is_similar(res1, res)


class TestLayerNorm:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            freeze=True, opt_level=OptLevel.level2
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0],
    )
    def test_slice_patterns(self, pattern_func):
        layernorm_test(self.xpu_graph_backend, pattern_func)


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(opt_level=OptLevel.level2)
    layernorm_test(xpu_graph_backend, fn0)
    layernorm_test(xpu_graph_backend, fn1)
