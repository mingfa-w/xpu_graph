import pytest
import random
import torch
import torch_mlu
import xpu_graph
from xpu_graph.config import OptLevel

from xpu_graph.test_utils import (
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)

device = "mlu:0"
data_type = torch.float16

def fn0(inputs, slice_, batch):
    zeros = torch.zeros([batch, 32], device=device, dtype=data_type)
    slice_1 = slice_[:, 0:32]
    slice_2 = slice_[:, 10118:10150]
    slice_3 = slice_[:, 10110:10142]
    slice_4 = slice_[:, 10579:10611]
    slice_5 = slice_[:, 10570:10602]
    slice_6 = slice_[:, 11032:11064]
    slice_7 = slice_[:, 11445:11477]
    slice_8 = slice_[:, 11440:11472]
    where_0 = torch.where(inputs, zeros, slice_2)
    where_1 = torch.where(inputs, zeros, slice_4)
    where_2 = torch.where(inputs, zeros, slice_6)
    where_3 = torch.where(inputs, zeros, slice_8)
    output = torch.cat([slice_1, where_0, slice_3, where_1, slice_5, where_2, slice_7, where_3], dim=-1)
    return output

def fn1(cond0, cond1, slice_, slice1_, batch):
    zeros = torch.zeros([batch, 32], device=device, dtype=data_type)
    slice_1 = slice_[:, 0:32]
    slice_2 = slice_[:, 10118:10150]
    slice_3 = slice_[:, 10110:10142]
    slice_4 = slice_[:, 10579:10611]
    slice_5 = slice_[:, 10570:10602]
    slice_6 = slice_[:, 11032:11064]
    slice_7 = slice_[:, 11445:11477]
    slice_8 = slice_[:, 11440:11472]

    where_0 = torch.where(cond0, zeros, slice_2)
    where_1 = torch.where(cond0, zeros, slice_4)
    where_2 = torch.where(cond1, zeros, slice_6)
    where_3 = torch.where(cond1, zeros, slice_8)

    slice_9 = slice1_[:, 0:32]
    slice_10 = slice1_[:, 10118:10150]
    slice_11 = slice1_[:, 10110:10142]
    slice_12 = slice1_[:, 10579:10611]
    slice_13 = slice1_[:, 10570:10602]
    slice_14 = slice1_[:, 11032:11064]
    slice_15 = slice1_[:, 11445:11477]
    slice_16 = slice1_[:, 11440:11472]

    where_4 = torch.where(cond0, zeros, slice_10)
    where_5 = torch.where(cond0, zeros, slice_12)
    where_6 = torch.where(cond1, zeros, slice_14)
    where_7 = torch.where(cond1, zeros, slice_16)

    output = torch.cat(
        [
            slice_1, where_0, slice_3, where_1,
            slice_5, where_2, slice_7, where_3,
            slice_9, where_4, slice_11, where_5,
            slice_13, where_6, slice_15, where_7
        ],
        dim=-1
    )
    return output

def fn2(inputs, slice_, batch):
    zeros = torch.zeros([batch, 4], device=device, dtype=data_type)
    slice_0 = slice_[:, 0:4]
    slice_1 = slice_[:, 4:8]
    slice_2 = slice_[:, 11984:11988]
    slice_3 = slice_[:, 11562:11566]
    slice_4 = slice_[:, 9782:9786]
    slice_5 = slice_[:, 9727:9731]
    slice_6 = slice_[:, 9745:9749]
    slice_7 = slice_[:, 9763:9767]
    slice_8 = slice_[:, 9773:9777]
    slice_9 = slice_[:, 9805:9809]
    slice_10 = slice_[:, 9822:9826]
    slice_11 = slice_[:, 10283:10287]
    slice_12 = slice_[:, 10736:10740]
    slice_13 = slice_[:, 10740:10744]
    where_0 = torch.where(inputs, zeros, slice_2)
    where_1 = torch.where(inputs, zeros, slice_3)
    where_2 = torch.where(inputs, zeros, slice_4)
    where_3 = torch.where(inputs, zeros, slice_5)
    where_4 = torch.where(inputs, zeros, slice_6)
    where_5 = torch.where(inputs, zeros, slice_7)
    where_6 = torch.where(inputs, zeros, slice_8)
    where_7 = torch.where(inputs, zeros, slice_9)
    where_8 = torch.where(inputs, zeros, slice_10)
    where_9 = torch.where(inputs, zeros, slice_11)
    where_10 = torch.where(inputs, zeros, slice_12)
    cat = torch.cat(
        [
            slice_0, where_0, slice_1, where_1, slice_1, where_2,
            where_3, where_4, slice_1, where_5, where_6, where_7,
            where_8, slice_13, where_9, where_10
        ],
        dim=-1
    )
    stack = torch.stack(
        [
            slice_0, where_0, slice_1, where_1, slice_1, where_2,
            where_3, where_4, slice_1, where_5, where_6, where_7,
            where_8, slice_13, where_9, where_10
        ],
        dim=0
    )
    return cat, stack

def fn3(inputs, slice_, batch):
    zeros = torch.zeros([batch, 4], device=device, dtype=data_type)
    ones = torch.ones([batch, 4], device=device, dtype=data_type)
    slice_1 = slice_[:, 0:4]
    slice_2 = slice_[:, 11984:11988]
    slice_3 = slice_[:, 11562:11566]
    slice_4 = slice_[:, 9782:9786]
    slice_5 = slice_[:, 9727:9731]
    slice_6 = slice_[:, 9745:9749]
    slice_7 = slice_[:, 9763:9767]
    slice_8 = slice_[:, 9773:9777]
    slice_9 = slice_[:, 9805:9809]
    slice_10 = slice_[:, 9822:9826]
    slice_11 = slice_[:, 10283:10287]
    slice_12 = slice_[:, 10736:10740]
    where_0 = torch.where(inputs, zeros, slice_2)
    where_1 = torch.where(inputs, zeros, slice_3)
    where_2 = torch.where(inputs, zeros, slice_4)
    where_3 = torch.where(inputs, zeros, slice_5)
    where_4 = torch.where(inputs, zeros, slice_6)
    where_5 = torch.where(inputs, zeros, slice_7)
    where_6 = torch.where(inputs, zeros, slice_8)
    where_7 = torch.where(inputs, zeros, slice_9)
    where_8 = torch.where(inputs, ones, slice_10)
    where_9 = torch.where(inputs, zeros, slice_11)
    where_10 = torch.where(inputs, zeros, slice_12)
    cat = torch.cat(
        [
            slice_1, where_0, where_1, where_2, slice_2, where_3,
            where_4, where_5, slice_3, where_6, where_7, where_8,
            slice_4, slice_5, where_9, where_10
        ],
        dim=-1
    )
    return cat, slice_1, slice_2, where_3

def fn4(inputs, slice_, batch):
    zeros = torch.zeros([batch, 32], device=device, dtype=data_type)
    slice_1 = slice_[:, 0:32]
    slice_2 = slice_[:, 10118:10150]
    slice_3 = slice_[:, 10110:10142]
    slice_4 = slice_[:, 10579:10611]
    slice_5 = slice_[:, 10570:10602]
    slice_6 = slice_[:, 11032:11064]
    slice_7 = slice_[:, 11445:11477]
    slice_8 = slice_[:, 11440:11472]
    where_0 = torch.where(inputs, zeros, slice_2)
    where_1 = torch.where(inputs, zeros, slice_4)
    where_2 = torch.where(inputs, zeros, slice_6)
    where_3 = torch.where(inputs, zeros, slice_8)
    output = torch.cat([slice_1, where_0, slice_3, where_1, slice_5, where_2, where_3], dim=-1)
    return output

def slice_where_test(xpu_graph_backend, func):
    batch = 512
    random_list = random.choices([0, 1], k=batch)
    cand = torch.tensor(random_list, device=device, dtype=data_type).unsqueeze(-1).bool()
    slice_ = torch.randn(batch, 35149, device=device, dtype=data_type)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)

    if func in [fn1]:
        cand1 = torch.tensor(random_list, device=device, dtype=data_type).unsqueeze(-1).bool()
        slice1_ = torch.randn(batch, 35149, device=device, dtype=data_type)
        res = func(cand, cand1, slice_, slice1_, batch)
        res1 = compiled(cand, cand1, slice_, slice1_, batch)
    else:
        res = func(cand, slice_, batch)
        res1 = compiled(cand, slice_, batch)

    for i in range(len(res)):
        assert torch.equal(res[i].cpu().float(), res1[i].cpu().float())

class TestSliceWhere:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(opt_level=OptLevel.level2, is_training=False)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn1,
            fn2,
            fn3,
            fn4,
        ],
    )
    def test_slice_where_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            slice_where_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.FusedSlice changed graph" in caplog.text
        assert "Pattern.FusedSliceWhere changed graph" in caplog.text
        if pattern_func in [fn2, fn3, fn4]:
            assert "Pattern.FusedSliceWhereCat changed graph" in caplog.text
        if pattern_func in [fn3]:
            assert "Pattern.FusedCatSlice changed graph" in caplog.text

if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(opt_level=OptLevel.level2, is_training=False)
    slice_where_test(xpu_graph_backend, fn0)
    slice_where_test(xpu_graph_backend, fn1)
    slice_where_test(xpu_graph_backend, fn2)
    slice_where_test(xpu_graph_backend, fn3)
    slice_where_test(xpu_graph_backend, fn4)
