import torch
import torch.nn as nn
import torch.nn.functional as F
from xpu_graph.config import OptLevel
import xpu_graph
from xpu_graph.test_utils import is_similar
import pytest

device = "mlu"
data_type = torch.float32


def fn0(arg0_1: torch.Tensor):
    # 默认值为 0
    mapped = torch.zeros_like(arg0_1, dtype=torch.int64, device=arg0_1.device)

    # 映射规则
    mapped = torch.where(arg0_1 == 1, torch.ones_like(mapped), mapped)
    mapped = torch.where(arg0_1 == 2, torch.full_like(mapped, 2), mapped)
    mapped = torch.where(arg0_1 == 3, torch.full_like(mapped, 3), mapped)
    mapped = torch.where(arg0_1 == 9998, torch.full_like(mapped, 4), mapped)
    mapped = torch.where(arg0_1 == 9999, torch.full_like(mapped, 5), mapped)
    return mapped
def fn1(arg0_1: torch.Tensor):
    # 默认值为 0
    mapped = torch.zeros_like(arg0_1, dtype=torch.int64, device=arg0_1.device)

    # 映射规则
    mapped = torch.where(arg0_1 == 1, 1, mapped)
    mapped = torch.where(arg0_1 == 2,2, mapped)
    mapped = torch.where(arg0_1 == 3, 3, mapped)
    mapped = torch.where(arg0_1 == 9998,4, mapped)
    mapped = torch.where(arg0_1 == 9999, 5, mapped)
    return mapped
def layernorm_test(xpu_graph, func):
    inputs = torch.randint(size=(8, 1024), low=0, high=9999,device=device, dtype=torch.int64)
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    res = func(inputs)
    res1 = compiled(inputs)
    assert is_similar(res, res1)


class TestLayerNormCast:
    def setup_class(self):
        config = xpu_graph.XpuGraphConfig(
            is_training=False, opt_level=OptLevel.level2, freeze=True
        )
        self.xpu_graph_backend = xpu_graph.XpuGraph(config)

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0],
    )
    def test_layernorm_cast_patterns(self, pattern_func):
        layernorm_test(self.xpu_graph_backend, pattern_func)


if __name__ == "__main__":
    config = xpu_graph.XpuGraphConfig(
        is_training=False, opt_level=OptLevel.level2, freeze=True, debug=True
    )
    xpu_graph_backend = xpu_graph.XpuGraph(config)
    layernorm_test(xpu_graph_backend, fn0)
