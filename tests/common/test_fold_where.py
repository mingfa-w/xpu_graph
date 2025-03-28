import pytest
import torch
import xpu_graph
from xpu_graph.test_utils import is_similar
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache

import torch

x = torch.rand(3, 3)
y = torch.rand(3, 3)
cond_rand = torch.rand(3, 3) > 0.5

ones = torch.ones_like(x)
zeros = torch.zeros_like(x)


# Case 0: cond is all True -> should be x
def fn0():
    full_true = torch.full_like(x, True, dtype=torch.bool)
    return torch.where(full_true, x, y)


# Case 1: cond is all False -> should be y
def fn1():
    full_false = torch.full_like(x, False, dtype=torch.bool)
    return torch.where(full_false, x, y)


# Case 2: x == y -> return x
def fn2():
    return torch.where(cond_rand, x, x)


# Case 3: torch.where(cond, ones, ones) -> ones
def fn3():
    ones = torch.ones_like(cond_rand)
    return torch.where(cond_rand, ones, ones)


# Case 4: torch.where(cond, zeros, zeros) -> zeros
def fn4():
    zeros = torch.zeros_like(cond_rand)
    return torch.where(cond_rand, zeros, zeros)


# Case 5: torch.where(cond, ones, zeros) -> cond
def fn5():
    ones = torch.ones_like(cond_rand)
    zeros = torch.zeros_like(cond_rand)
    return torch.where(cond_rand, ones, zeros)


# Case 6: torch.where(cond, ones, y) -> torch.where(cond, cond, y)
def fn6():
    ones = torch.ones_like(cond_rand)
    return torch.where(cond_rand, ones, y)


# Case 7: torch.where(cond, x, zeros) -> torch.where(cond, x, cond)
def fn7():
    zeros = torch.zeros_like(cond_rand)
    return torch.where(cond_rand, x, zeros)


def where_test(xpu_graph, func):
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    res1 = func()
    res = compiled()
    assert is_similar(res1.float(), res.float())


class TestWhere:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn1,
            fn2,
            fn3,
            fn4,
            fn5,
            fn6,
            fn7,
        ],
    )
    def test_where_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph):
            where_test(self.xpu_graph, pattern_func)
            assert "Pattern.FoldWhere changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=True)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    where_test(xpu_graph, fn0)
    where_test(xpu_graph, fn1)
    where_test(xpu_graph, fn2)
    where_test(xpu_graph, fn3)
    where_test(xpu_graph, fn4)
    where_test(xpu_graph, fn5)
