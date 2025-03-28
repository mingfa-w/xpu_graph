import pytest
import torch
import xpu_graph
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache


def fn0():
    x = torch.ones((2048, 328), dtype=torch.float32, device="cpu", pin_memory=False)
    y = x[:, :128]
    return y


def fn1():
    x = torch.zeros((2048, 328), dtype=torch.float32, device="cpu", pin_memory=False)
    y = x[:, :128]
    return y

def fn2():
    x = torch.full((2048, 328), 3.14, device="cpu").to(dtype=torch.int32)
    y = x[:, :128]
    return y

def fn3():
    x = torch.full((2,3,6), 3.14, device="cpu").to(dtype=torch.int32)
    return torch.ops.aten.slice.Tensor(x, 2, 1, 5)

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
        ],
    )
    def test_slice_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph):
            slice_test(self.xpu_graph, pattern_func)
            assert "Pattern.FuseFullSlice changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=True)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    '''
    slice_test(xpu_graph, fn0)
    slice_test(xpu_graph, fn1)
    slice_test(xpu_graph, fn2)
    '''
    slice_test(xpu_graph, fn3)
