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

def fn0(x1, x2, x3, x4):
    a = x1 * x2
    sum_a = a.sum(dim=1)
    b = x3 * x4
    sum_b = b.sum(dim=1)
    out = torch.cat([sum_a, sum_b], dim=1)
    return out

def bmm_test(xpu_graph_backend, func):
    batch = 2048
    dtype = torch.half
    a = torch.randn(batch, 128, 128, dtype=dtype, device="mlu")
    b = torch.randn(batch, 128, 128, dtype=dtype, device="mlu")
    c = torch.randn(batch, 128, 128, dtype=dtype, device="mlu")
    d = torch.randn(batch, 128, 128, dtype=dtype, device="mlu")

    res1 = func(a,b,c,d)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res = compiled(a,b,c,d)

    assertTensorsEqual(
        res.cpu().float(), res1.cpu().float(), 0.005, use_MSE=True, use_RAE=True
    )


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False, freeze=True, opt_level=OptLevel.level2, vendor_compiler_config=False, debug = True
    )
    bmm_test(xpu_graph_backend, fn0)
