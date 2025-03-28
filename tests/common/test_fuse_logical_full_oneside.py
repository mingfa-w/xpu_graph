import pytest
import torch
import xpu_graph
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache


def fn0():
    x = torch.ones((128, 64), dtype=torch.float32, device="cpu", pin_memory=False)
    return x == 1.0


def fn1():
    x = torch.zeros((128, 64), dtype=torch.float32, device="cpu", pin_memory=False)
    return x == 0.0


def fn2():
    x = torch.ones((128, 64), dtype=torch.float32, device="cpu", pin_memory=False)
    return x != 1.0


def fn3():
    x = torch.zeros((128, 64), dtype=torch.float32, device="cpu", pin_memory=False)
    return x != 0.0


def fn4():
    x = torch.zeros((128, 64), dtype=torch.float32, device="cpu", pin_memory=False)
    return x == 1.0


def fn5():
    x = torch.ones((128, 64), dtype=torch.float32, device="cpu", pin_memory=False)
    return x != 0.0


def fn6():
    x = torch.full((128, 64), 0.0, dtype=torch.float32)
    return x != 0.0


def slice_test(xpu_graph, func):
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    res = func()
    res1 = compiled()
    assert torch.equal(res.float(), res.float())


class TestOneSlice:
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
        ],
    )
    def test_slice_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph):
            slice_test(self.xpu_graph, pattern_func)
            assert "FuseLogicalFullOneSide" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=True, debug=True)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    slice_test(xpu_graph, fn0)
    slice_test(xpu_graph, fn1)
    slice_test(xpu_graph, fn2)
    slice_test(xpu_graph, fn3)
    slice_test(xpu_graph, fn4)
    slice_test(xpu_graph, fn5)
    slice_test(xpu_graph, fn6)
    slice_test(xpu_graph, fn7)
    slice_test(xpu_graph, fn8)
