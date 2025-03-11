import pytest
import random
import torch
import torch_mlu
import xpu_graph
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import assertTensorsEqual

device = "mlu:0"
data_type = torch.float16
aten = torch.ops.aten

def fn0(where, inputs):
    zeros = torch.zeros([64, 32], device=device, dtype=data_type)
    slice_0 = inputs[:,0:32]
    slice_1 = inputs[:,32:64]
    slice_2 = inputs[:,64:96]
    slice_3 = inputs[:,96:128]
    where_0 = torch.where(where, zeros, slice_0)
    where_1 = torch.where(where, zeros, slice_1)
    where_2 = torch.where(where, zeros, slice_2)
    where_3 = torch.where(where, zeros, slice_3)
    inputs_list = [where_0, where_1, where_2, where_3]
    stack = torch.stack(inputs_list, dim=0)
    return stack

def stack_to_cat_test(xpu_graph_backend, func):
    random_list = random.choices([0, 1], k=64)
    where = torch.tensor(random_list, device=device, dtype=data_type).unsqueeze(-1).bool()
    inputs = torch.randn(64, 35149, device=device, dtype=data_type)

    res = func(where, inputs)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = compiled(where, inputs)
    assertTensorsEqual(
        res[0].cpu().float(), res1[0].cpu().float(), 0.0001, use_MSE=True, use_RAE=True
    )

class TestStackToCat:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(opt_level=OptLevel.level2)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
        ],
    )
    def test_stack_to_cat_patterns(self, pattern_func):
        stack_to_cat_test(self.xpu_graph_backend, pattern_func)

if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(opt_level=OptLevel.level2)
    stack_to_cat_test(xpu_graph_backend, fn0)

