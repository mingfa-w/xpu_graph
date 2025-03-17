import torch
import torch.nn as nn
import torch.nn.functional as F
from xpu_graph.config import OptLevel
import xpu_graph
from xpu_graph.test_utils import is_similar, maybe_similar
import pytest

device = "cpu"
data_type = torch.float32
aten = torch.ops.aten


def fn0(indices, values):
    max_len = torch.max(indices) + 1
    result = torch.zeros(max_len, dtype=values.dtype, device=values.device)
    result.scatter_(0, indices, values)
    return result


def inplace_test(xpu_graph, func):
    indices = torch.tensor([7, 6, 5, 4, 0, 1, 2, 3], dtype=torch.int64, device=device)
    values = torch.randn([8], dtype=data_type, device=device)
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)

    res1 = func(indices, values)
    res = compiled(indices, values)
    assert is_similar(res, res1)


class TestInplace:
    def setup_class(self):
        infer_config = xpu_graph.XpuGraphConfig(
            is_training=False, opt_level=OptLevel.level2
        )
        self.infer_backend = xpu_graph.XpuGraph(infer_config)

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0],
    )
    def test_inplace(self, pattern_func):
        inplace_test(self.infer_backend, pattern_func)


if __name__ == "__main__":
    infer_config = xpu_graph.XpuGraphConfig(
        is_training=False, opt_level=OptLevel.level2, debug=True
    )
    infer_backend = xpu_graph.XpuGraph(infer_config)
    inplace_test(infer_backend, fn0)
