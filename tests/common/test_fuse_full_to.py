import pytest
import torch
import xpu_graph
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache


def fn0():
    x = torch.ones((4, 4), device="cpu").to(dtype=torch.float32)
    return x


def fn1():
    y = torch.zeros((4, 4), device="cpu").to(dtype=torch.bool)
    return y


def fn2():
    z = torch.full((4, 4), 3.14, device="cpu").to(dtype=torch.int32)
    return z


def to_test(xpu_graph, func):
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
        ],
    )
    def test_to_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph):
            to_test(self.xpu_graph, pattern_func)
            assert "Pattern.FuseFullTo changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=True, debug=True)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    to_test(xpu_graph, fn0)
    to_test(xpu_graph, fn1)
    to_test(xpu_graph, fn2)
