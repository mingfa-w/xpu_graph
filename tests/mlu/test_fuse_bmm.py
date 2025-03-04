import pytest
import torch
import torch_mlu
import xpu_graph
from xpu_graph.config import OptLevel
import torch.nn.functional as F
from xpu_graph.test_utils import assertTensorsEqual

device = "mlu:0"
data_type = torch.float16
aten = torch.ops.aten


def fn0(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight)
    return output + bias if bias is not None else output

def fn1(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight)
    output = output + bias
    return output

def fn2(inputs, weight, residual):
    output = torch.matmul(inputs, weight)
    output = output + residual
    return output

def fn3(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight)
    output = output
    output = F.gelu(output)
    return output

def fn4(inputs, weight, bias=None):
    output = torch.matmul(inputs, weight)
    output = output + bias
    output = F.gelu(output)
    return output

def fn5(inputs, weight, residual):
    output = torch.matmul(inputs, weight)
    output = output + residual
    output = F.gelu(output)
    return output

def matmul_test(xpu_graph_backend, func):
    input_a = torch.randn((5, 32, 128), device=device, dtype=data_type)
    input_b = torch.randn((5, 128, 256), device=device, dtype=data_type)
    if func in [fn0, fn3]:
        bias = None
    if func in [fn1, fn4]:
        bias = torch.randn((5, 1, 256), device=device, dtype=data_type)
    if func in [fn2, fn5]:
        residual = torch.randn((5, 32, 256), device=device, dtype=data_type)
        bias = residual

    res = func(input_a, input_b, bias)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = compiled(input_a, input_b, bias)
    assertTensorsEqual(
        res.cpu().float(), res1.cpu().float(), 0.005, use_MSE=True, use_RAE=True
    )


class TestMatMul:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(opt_level=OptLevel.level2)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn1,
            fn2,
            fn3,
            fn4,
            fn5,
        ],
    )
    def test_matmul_patterns(self, pattern_func):
        matmul_test(self.xpu_graph_backend, pattern_func)

if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(opt_level=OptLevel.level2)
    matmul_test(xpu_graph_backend, fn0)
    matmul_test(xpu_graph_backend, fn1)
    matmul_test(xpu_graph_backend, fn2)
    matmul_test(xpu_graph_backend, fn3)
    matmul_test(xpu_graph_backend, fn4)
    matmul_test(xpu_graph_backend, fn5)
