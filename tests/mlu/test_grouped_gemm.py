import pytest
import torch
import torch_mlu
import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    assertTensorsEqual,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)

device = "mlu:0"
data_type = torch.float16


def fn0(inputs, weight_list):
    outputs_original = []
    for weight in weight_list:
        outputs_original.append(inputs @ weight)
    return outputs_original


def combine_matmul_test(xpu_graph_backend, func):
    inputs = torch.randn((20, 30), device=device, dtype=data_type)
    #input2 = torch.randn((20, 30), device=device, dtype=data_type)
    #input3 = torch.randn((20, 30), device=device, dtype=data_type)
    #input4 = torch.randn((20, 30), device=device, dtype=data_type)
    #input5 = torch.randn((20, 30), device=device, dtype=data_type)
    #input_list = [input1, input2, input3, input4, input5]
    weight1 = torch.randn((30, 40), device=device, dtype=data_type)
    weight2 = torch.randn((30, 40), device=device, dtype=data_type)
    weight3 = torch.randn((30, 40), device=device, dtype=data_type)
    weight4 = torch.randn((30, 40), device=device, dtype=data_type)
    weight5 = torch.randn((30, 40), device=device, dtype=data_type)
    weight_list = [weight1, weight2, weight3, weight4, weight5]
    res = func(inputs, weight_list)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = compiled(inputs, weight_list)
    for i in range(5):
        assertTensorsEqual(
            res[i].cpu().float(),
            res1[i].cpu().float(),
            0.005,
            use_MSE=True,
            use_RAE=True,
        )


class TestCombineMatMul:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False, opt_level=OptLevel.level2
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
        ],
    )
    def test_matmul_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            combine_matmul_test(self.xpu_graph_backend, pattern_func)
            assert "Pattern.FusedCombineMatMul changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False, opt_level=OptLevel.level2, debug=False, vendor_compiler_config=None
    )
    combine_matmul_test(xpu_graph_backend, fn0)

'''
from apex.contrib import grouped_gemm
inputs = torch.randn((100, 30), device=device, dtype=data_type)
weight1 = torch.randn((30, 40), device=device, dtype=data_type)
weight2 = torch.randn((30, 40), device=device, dtype=data_type)
weight3 = torch.randn((30, 40), device=device, dtype=data_type)
weight4 = torch.randn((30, 40), device=device, dtype=data_type)
weight5 = torch.randn((30, 40), device=device, dtype=data_type)
weight_list = [weight1, weight2, weight3, weight4, weight5]
weights = torch.stack(weight_list, dim=0)
print("weights.shape: ", weights.shape)
batch_list = [20, 20, 20, 20, 20]
batch_tensor = torch.tensor(
    batch_list, dtype=torch.int64, device="cpu",
)
output = grouped_gemm.ops.gmm(inputs, weights, batch_tensor, trans_b=False)
print("output.shape: ", output.shape)
output = output.view(len(batch_list), batch_list[0], -1)
print("output.shape: ", output.shape)
'''
