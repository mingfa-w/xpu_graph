import math
import pytest

import torch
import xpu_graph

from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    assertTensorsEqual,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)


def fn0():
    x = torch.zeros([], dtype=torch.float32, device='mlu')
    y = torch.zeros([], dtype=torch.float32, device='mlu')
    z = torch.zeros([], dtype=torch.float32, device='mlu')
    return x,y,z 

def fn1():
    x = torch.zeros([2048], dtype=torch.float32, device='mlu')
    y = torch.zeros([2048], dtype=torch.float32, device='mlu')
    z = torch.zeros([2048], dtype=torch.float32, device='mlu')
    return x,y,z 

def gather_test(xpu_graph_backend, func):
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = func()
    res = compiled()
    print(res)
    print(res1)


class TestGather:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False, debug=True, opt_level=OptLevel.level1
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0],
    )
    def test_gather_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            gather_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.FusedGatherToCopy changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=True, debug=False, opt_level=OptLevel.level1
    )
    gather_test(xpu_graph_backend, fn1)
    gather_test(xpu_graph_backend, fn0)
    '''

    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False, debug=False, opt_level=OptLevel.level1
    )
    gather_test(xpu_graph_backend, fn1)
    gather_test(xpu_graph_backend, fn0)
    '''
