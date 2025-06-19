import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import xpu_graph
from xpu_graph import XpuGraph, XpuGraphConfig
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    assertTensorsEqual,
    is_similar,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)

device = "cpu"
data_type = torch.float
aten = torch.ops.aten


def fn0(inputs, weight, bias, scale, q_weight, q_bias):
    inputs = inputs * scale
    origin_dtype = inputs.dtype
    inputs = inputs.float()
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, correction=False)
    normalized = (inputs - mean) * ((variance + 1e-5) ** (-0.5))
    outputs = weight * normalized + bias
    outputs = outputs.to(dtype=origin_dtype)
    return outputs


def layernorm_test(xpu_graph, func):
    inputs = torch.randn((64, 1362), device=device, dtype=data_type)
    residual = torch.randn((64, 1362), device=device, dtype=data_type)
    weight = torch.randn((1362), device=device, dtype=data_type)
    bias = torch.randn((1362,), device=device, dtype=data_type)
    q_weight = torch.randn((1362, 1362), device=device, dtype=data_type)
    scale = 0.1
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)

    q_bias = None
    norm = compiled(inputs, weight, bias, scale, q_weight, q_bias)
    norm1 = func(inputs, weight, bias, scale, q_weight, q_bias)
    assertTensorsEqual(norm1.cpu().float(), norm.cpu().float(), 0.005, use_MSE=True, use_RAE=True)


class TestLayerNorm:
    def setup_class(self):
        self.xpu_graph_backend = XpuGraph(XpuGraphConfig(is_training=False, freeze=True, opt_level=OptLevel.level2))

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0],
    )
    def test_layernorm_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            layernorm_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.FusedScaleLayernorm changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = XpuGraph(XpuGraphConfig(is_training=False, freeze=True, opt_level=OptLevel.level2, debug=True))
    layernorm_test(xpu_graph_backend, fn0)
