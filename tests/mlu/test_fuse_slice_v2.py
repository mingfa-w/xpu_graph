import pytest

import torch
import torch_mlu
import xpu_graph
from xpu_graph import OptLevel
from xpu_graph.test_utils import (
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)


device = "mlu:0"
aten = torch.ops.aten


def fn0(x):
    axis = -1
    return  x[:, 34691:34754] , x[:, 38371:38433] , x[:, 41479:41540]

def fn1(x):
    axis = -1
    return  x[:, 34691:34754] , x[:, 38371:38433] , x[:, 41479:41540], x[:, 41379:41389]


def slice_test(xpu_graph_backend, func):
    for batch in (10, ):#512, 256, 128, 64, 32):
        a = torch.randn(batch, 43106).to(device=device)
        compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
        res = compiled(a)[0]
        res1 = func(a)[0]
        assert torch.equal(res.cpu().float(), res1.cpu().float())


class TestSliceV2:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False, freeze=True, opt_level=OptLevel.level1
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
        ],
    )
    def test_slice_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            slice_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.FusedSlice changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False, freeze=True, opt_level=OptLevel.level1, debug=True
    )
    slice_test(xpu_graph_backend, fn0)
    slice_test(xpu_graph_backend, fn1)
