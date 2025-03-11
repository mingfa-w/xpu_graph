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

def fn0(inputs, slice_):
    zeros = torch.zeros([64, 32], device=device, dtype=data_type)
    slice_0 = slice_[:, 0:4]
    slice_2 = slice_[:, 10118:10150]
    slice_4 = slice_[:, 10579:10611]
    slice_5 = slice_[:, 11032:11064]
    slice_6 = slice_[:, 11445:11477]
    where_0 = torch.where(inputs, zeros, slice_2)
    where_1 = torch.where(inputs, zeros, slice_4)
    where_2 = torch.where(inputs, zeros, slice_5)
    where_3 = torch.where(inputs, zeros, slice_6)
    output = torch.cat([slice_0, where_0, where_1, where_2, where_3], dim=-1)
    return output, slice_0



def where_slice_cat_test(xpu_graph_backend, func):
    random_list = random.choices([0, 1], k=64)
    inputs = torch.tensor(random_list, device=device, dtype=data_type).unsqueeze(-1).bool()
    slice_ = torch.randn(64, 35149, device=device, dtype=data_type)

    res = func(inputs, slice_)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res1 = compiled(inputs, slice_)
    assertTensorsEqual(
        res[0].cpu().float(), res1[0].cpu().float(), 0.0001, use_MSE=True, use_RAE=True
    )

class TestWhereSliceCat:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(opt_level=OptLevel.level2)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
        ],
    )
    def test_where_cat_patterns(self, pattern_func):
        where_slice_cat_test(self.xpu_graph_backend, pattern_func)

if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(opt_level=OptLevel.level2)
    where_slice_cat_test(xpu_graph_backend, fn0)
