import math
import pytest
import torch
import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import assertTensorsEqual
from torch import nn
from torch.nn import functional as F

def fn0(inputs, weight, bias = None , residual= None,alpha = 1,beta = 1):
    weight = weight.transpose(0,1)
    proj = torch.matmul(inputs,weight)
    if bias is not None:
        proj = torch.matmul(inputs,weight) + bias
        if residual is not None:
            proj = alpha * (torch.matmul(inputs,weight) + bias) + beta * residual
    return proj
    


def mmadd_test(xpu_graph_backend, func):
    dtype = torch.half
    N, T, input_size, hidden_size, alpha, beta = 32, 129, 2048, 4096, 0.5, 0.1
    input = torch.randn(N,T,input_size , dtype=dtype, device="mlu")
    weight = torch.randn(hidden_size*3, input_size , dtype=dtype, device="mlu")
    bias = torch.randn(hidden_size * 3, dtype=dtype, device="mlu")
    residual = torch.randn(N,T,hidden_size * 3, dtype=dtype, device="mlu")

    res_mm1 = func(input, weight)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res_mm = compiled(input,weight)
    res_mm_bias1 = func(input, weight,bias)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res_mm_bias = compiled(input,weight,bias)
    

    res_mm_bias_residual1 = func(input, weight,bias,residual)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res_mm_bias_residual = compiled(input,weight,bias,residual)


    assertTensorsEqual(
        res_mm.cpu().float(), res_mm1.cpu().float(), 0.005, use_MSE=True, use_RAE=True
    )
    assertTensorsEqual(
        res_mm_bias.cpu().float(), res_mm_bias1.cpu().float(), 0.005, use_MSE=True, use_RAE=True
    )

    assertTensorsEqual(
        res_mm_bias_residual.cpu().float(), res_mm_bias_residual1.cpu().float(), 0.005, use_MSE=True, use_RAE=True
    )

class TestMMADD:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig()
        config.target = xpu_graph.config.Target.mlu
        config.opt_level = OptLevel.level2
        self.xpu_graph_backend = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0],
    )
    def test_sfdp_patterns(self, caplog, pattern_func):
        mmadd_test(self.xpu_graph_backend, pattern_func)
        #assert "Pattern.FusedBMM changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig()
    config.target = xpu_graph.config.Target.mlu
    config.opt_level = OptLevel.level2
    # config.vendor_compiler = {"mode": "reduce-overhead"}
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    mmadd_test(xpu_graph, fn0)