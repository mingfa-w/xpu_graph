import pytest
import torch
import math

import torch
import xpu_graph

from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    assertTensorsEqual,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)


def fn0(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    sum_5 = torch.sum(x1, dim=-1)
    sum_8 = torch.sum(x2, dim=-1)
    sum_11 = torch.sum(x3, dim=-1)
    sum_14 = torch.sum(x4, dim=-1)
    sum_17 = torch.sum(x5, dim=-1)
    sum_20 = torch.sum(x6, dim=-1)
    sum_23 = torch.sum(x7, dim=-1)
    sum_26 = torch.sum(x8, dim=-1)
    sum_29 = torch.sum(x9, dim=-1)
    sum_32 = torch.sum(x10, dim=-1)
    sum_115 = torch.sum(x11, dim=-1)

    result = torch.stack(
        [
            sum_5,
            sum_8,
            sum_11,
            sum_14,
            sum_17,
            sum_20,
            sum_23,
            sum_26,
            sum_29,
            sum_32,
            sum_115,
        ]
    )
    return result


def fn1(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    sum_5 = torch.sum(x1, dim=-1)
    sum_8 = torch.sum(x2, dim=-1)
    sum_11 = torch.sum(x3, dim=-1)
    sum_14 = torch.sum(x4, dim=-1)
    sum_17 = torch.sum(x5, dim=-1)
    sum_20 = torch.sum(x6, dim=-1)
    sum_23 = torch.sum(x7, dim=-1)
    sum_26 = torch.sum(x8, dim=-1)
    sum_29 = torch.sum(x9, dim=-1)
    sum_32 = torch.sum(x10, dim=-1)
    sum_115 = torch.sum(x11, dim=-1)

    result = torch.cat(
        [
            sum_5,
            sum_8,
            sum_11,
            sum_14,
            sum_17,
            sum_20,
            sum_23,
            sum_26,
            sum_29,
            sum_32,
            sum_115,
        ],
        dim=-1,
    )
    return result


def fn2(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11):
    sum_5 = torch.sum(x1, dim=[1], keepdim=True)
    sum_8 = torch.sum(x2, dim=[1], keepdim=True)
    sum_11 = torch.sum(x3, dim=[1], keepdim=True)
    sum_14 = torch.sum(x4, dim=[1], keepdim=True)
    sum_17 = torch.sum(x5, dim=[1], keepdim=True)
    sum_20 = torch.sum(x6, dim=[1], keepdim=True)
    sum_23 = torch.sum(x7, dim=[1], keepdim=True)
    sum_26 = torch.sum(x8, dim=[1], keepdim=True)
    sum_29 = torch.sum(x9, dim=[1], keepdim=True)
    sum_32 = torch.sum(x10, dim=[1], keepdim=True)
    sum_115 = torch.sum(x11, dim=[1], keepdim=True)

    result = torch.cat(
        [
            sum_5,
            sum_8,
            sum_11,
            sum_14,
            sum_17,
            sum_20,
            sum_23,
            sum_26,
            sum_29,
            sum_32,
            sum_115,
        ],
        dim=-1,
    )
    return result


def mul_sum_stack_test(xpu_graph_backend, func):
    device = "mlu"
    example_inputs = [
        torch.randn(112, 64).to(device),
        torch.randn(112, 32).to(device),
        torch.randn(112, 16).to(device),
        torch.randn(112, 16).to(device),
        torch.randn(112, 16).to(device),
        torch.randn(112, 16).to(device),
        torch.randn(112, 16).to(device),
        torch.randn(112, 16).to(device),
        torch.randn(112, 16).to(device),
        torch.randn(112, 16).to(device),
        torch.randn(112, 8).to(device),
    ]

    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = func(*example_inputs)
    res = compiled(*example_inputs)
    assertTensorsEqual(
        res.cpu().float(), res1.cpu().float(), 0.001, use_MSE=True, use_RAE=True
    )


"""
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
        assert "Pattern.FusedMulCat changed graph" in caplog.text

"""

if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False,
    )
    # mul_sum_stack_test(xpu_graph_backend, fn0)
    mul_sum_stack_test(xpu_graph_backend, fn1)
    mul_sum_stack_test(xpu_graph_backend, fn2)
