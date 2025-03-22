import pytest
import torch
import torch_mlu
import xpu_graph
from xpu_graph.config import OptLevel
import torch.nn.functional as F
from xpu_graph.test_utils import assertTensorsEqual

device = "mlu:0"
data_type = torch.float32
aten = torch.ops.aten


def fn0(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight.t())
    return output


def fn1(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight.t()) + bias
    return output


def matmul_test(xpu_graph_backend, func):
    inputs = torch.randn((4096, 768), device=device, dtype=data_type)
    weight = torch.randn((16, 768), device=device, dtype=data_type)
    bias = torch.randn((16), device=device, dtype=data_type)
    res = func(inputs, weight, bias)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = compiled(inputs, weight, bias)
    assertTensorsEqual(
        res.cpu().float(), res1.cpu().float(), 0.005, use_MSE=True, use_RAE=True
    )


class TestMatMulToLinear:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False, opt_level=OptLevel.level2
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn1,
        ],
    )
    def test_matmul_patterns(self, pattern_func):
        matmul_test(self.xpu_graph_backend, pattern_func)


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=True, opt_level=OptLevel.level2, debug=True 
    )
