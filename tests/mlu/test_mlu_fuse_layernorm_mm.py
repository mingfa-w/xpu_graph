import torch
import torch.nn as nn
import torch.nn.functional as F
from xpu_graph.config import OptLevel
import xpu_graph
from xpu_graph.test_utils import is_similar
import pytest
from xpu_graph.test_utils import assertTensorsEqual

device = "mlu:0"
data_type = torch.float16
aten = torch.ops.aten

def fn0(inputs, residual, weight, bias,q_weight,q_bias=None,k_weight=None,k_bias=None,v_weight=None,v_bias=None,norm_out: bool = False):
    #inputs_ = inputs + residual
    inputs_ = inputs
    normed = torch.layer_norm(
        inputs_, normalized_shape=[512], weight=weight, bias=bias, eps=1e-5
    )
    
    proj_q = torch.matmul(normed,  q_weight.transpose(1, 0))
    if q_bias is not None:
        proj_q = proj_q + q_bias
    if k_weight is not None:
        proj_k = torch.matmul(normed, k_weight.transpose(1,0))
        if k_bias is not None:
            proj_k = proj_k + k_bias 
    if v_weight is not None:
        proj_v = torch.matmul(normed, v_weight.transpose(1,0)) + v_bias
        if v_bias is not None:
            proj_v = proj_v + v_bias
    #tuple
    outputs = [proj_q]
    if k_weight is not None:
        outputs.append(proj_k)
    if v_weight is not None:
        outputs.append(proj_v)
    if norm_out:
        outputs.append(normed)
    return tuple(outputs)


def layernorm_mul_test(xpu_graph, func):
    N, T, input_size, hidden_size = 4, 16, 512, 768
    inputs = torch.randn(N, T, input_size, dtype=data_type, device="mlu")
    weight = torch.randn(hidden_size * 3, input_size, dtype=data_type, device="mlu")
    bias = torch.randn(hidden_size * 3, dtype=data_type, device="mlu")
    norm_weight = torch.randn(input_size, dtype=data_type, device="mlu")
    norm_bias = torch.randn(input_size, dtype=data_type, device="mlu")
    residual = torch.randn(N, T, hidden_size, dtype=data_type, device="mlu")
    weights = torch.chunk(weight, 3)
    biass = torch.chunk(bias, 3)
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    res_with_qbias = compiled(inputs, residual, norm_weight, norm_bias,weights[0],biass[0])
    res1_with_qbias = func(inputs, residual, norm_weight, norm_bias,weights[0],biass[0])
    res = compiled(inputs, residual, norm_weight, norm_bias,weights[0],biass[0],weights[1],biass[1],weights[2],biass[2])
    res1 = func(inputs, residual, norm_weight, norm_bias,weights[0],biass[0],weights[1],biass[1],weights[2],biass[2])
    res = compiled(inputs, residual, norm_weight, norm_bias,weights[0])
    res1 = func(inputs, residual, norm_weight, norm_bias,weights[0])
    assertTensorsEqual(
        res_with_qbias[0].cpu().float(), res1_with_qbias[0].cpu().float(), 0.005, use_MSE=True, use_RAE=True
    )
    assertTensorsEqual(
        res[0].cpu().float(), res1[0].cpu().float(), 0.005, use_MSE=True, use_RAE=True
    )

class TestLayerNorm:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig()
        config.target = xpu_graph.config.Target.mlu
        config.opt_level = OptLevel.level2
        self.xpu_graph_backend = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0],
    )
    def test_slice_patterns(self, pattern_func):
        layernorm_mul_test(self.xpu_graph_backend, pattern_func)


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig()
    config.target = xpu_graph.config.Target.mlu
    config.opt_level = OptLevel.level2
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    layernorm_mul_test(xpu_graph, fn0)