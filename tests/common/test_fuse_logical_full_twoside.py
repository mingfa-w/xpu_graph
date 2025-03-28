import pytest
import torch
import xpu_graph
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache


def fn0():
    a = torch.full((4, 4), False, dtype=torch.bool)
    b = torch.full((4, 4), True, dtype=torch.bool)
    return torch.logical_or(a, b)


def fn1():
    a = torch.zeros((4, 4), dtype=torch.bool)
    b = torch.full((4, 4), True, dtype=torch.bool)
    return torch.logical_or(a, b)


def fn2():
    a = torch.zeros((4, 4), dtype=torch.bool)
    b = torch.ones((4, 4), dtype=torch.bool)
    return torch.logical_or(a, b)


def fn3():
    a = torch.full((4, 4), False, dtype=torch.bool)
    b = torch.full((4, 4), True, dtype=torch.bool)
    return torch.logical_and(a, b)


def fn4():
    a = torch.zeros((4, 4), dtype=torch.bool)
    b = torch.full((4, 4), True, dtype=torch.bool)
    return torch.logical_and(a, b)


def fn5():
    a = torch.zeros((4, 4), dtype=torch.bool)
    b = torch.ones((4, 4), dtype=torch.bool)
    return a == b


def fn6():
    a = torch.full((4, 4), False, dtype=torch.bool)
    b = torch.full((4, 4), True, dtype=torch.bool)
    return a != b


def fn7():
    a = torch.zeros((4, 4), dtype=torch.bool)
    b = torch.full((4, 4), True, dtype=torch.bool)
    return a == b


def fn8():
    a = torch.zeros((4, 4), dtype=torch.bool)
    b = torch.ones((4, 4), dtype=torch.bool)
    return torch.logical_and(a, b)


def logical_test(xpu_graph, func):
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    res = func()
    res1 = compiled()
    assert torch.equal(res.float(), res.float())


class TestFullTo:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=True)
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
            fn8,
        ],
    )
    def test_logical_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph):
            logical_test(self.xpu_graph, pattern_func)
            assert "FuseLogicalFullTwoSide" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=True, debug=True)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    logical_test(xpu_graph, fn0)
    logical_test(xpu_graph, fn1)
    logical_test(xpu_graph, fn2)
    logical_test(xpu_graph, fn3)
    logical_test(xpu_graph, fn4)
    logical_test(xpu_graph, fn5)
    logical_test(xpu_graph, fn6)
    logical_test(xpu_graph, fn7)
    logical_test(xpu_graph, fn8)
