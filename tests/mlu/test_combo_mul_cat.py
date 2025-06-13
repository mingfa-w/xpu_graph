import pytest
import math

import torch
import xpu_graph

from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    assertTensorsEqual,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)


def fn0(tensors):
    a, b, c, d, e, f, g, h = tensors
    result = torch.cat([a * b, c * d, e * f, g * h], dim=1)
    return result, 1


def fn1(tensors):
    a, b, c, d, e, f, g, h = tensors
    mul1 = a * b
    mul2 = c * d
    mul3 = e * f
    mul4 = g * h
    result = torch.cat([mul1, mul2, mul3, mul4], dim=1)
    return result, mul1


def fn2(tensors):
    a, b, c, d, e, f, g, h = tensors
    mul1 = a * b
    mul2 = c * d
    mul3 = e * f
    mul4 = g * h
    mul5 = g * 0.1
    result = torch.cat([mul1, mul2, mul3, mul4, mul5], dim=1)
    return result, mul1


def fn3(tensors):
    a, b, c, d, e, f, g, h = tensors
    mul1 = a * 0.4
    mul2 = c * 0.4
    mul3 = e * 0.4
    mul4 = b * 0.4
    mul5 = g * 0.1
    result = torch.cat([mul1, mul2, mul3, mul4, mul5], dim=1)
    return result, mul1


def mul_sum_cat_test(xpu_graph_backend, func):
    batch = 1024
    dtype = torch.half
    device = "mlu"
    batch_size, feature_dim = 3, 4
    tensors = [torch.randn(batch_size, feature_dim).to(device) for i in range(8)]

    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = func(tensors)
    res = compiled(tensors)
    assertTensorsEqual(
        res[0].cpu().float(), res1[0].cpu().float(), 0.001, use_MSE=True, use_RAE=True
    )


class TestMulCat:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn1,
            fn2,
        ],
    )
    def test_mul_cat_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            mul_sum_cat_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.ComboMulCat changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False,
    )
    # mul_sum_cat_test(xpu_graph_backend, fn0)
    # mul_sum_cat_test(xpu_graph_backend, fn1)
    # mul_sum_cat_test(xpu_graph_backend, fn2)
    mul_sum_cat_test(xpu_graph_backend, fn3)
