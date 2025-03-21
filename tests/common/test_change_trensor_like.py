import pytest
import torch
import xpu_graph
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache


def fn0(a):
    output = torch.ones_like(a)
    return output

def fn1(a):
    output = torch.zeros_like(a)
    return output

def fn2(a):
    output = torch.ones_like(a)
    return output

def tensorlike_test(xpu_graph, func):
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    a = torch.randn(128, 64)
    res = func(a)
    res1 = compiled(a)
    for i in range(len(res)):
        assert torch.equal(res[i].float(), res1[i].float())


class TestTensorLike:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
            fn1,
        ],
    )
    def test_tensorlike_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph):
            tensorlike_test(self.xpu_graph, pattern_func)
        assert "Pattern.ChangeTensorLike changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False, debug=True)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    tensorlike_test(xpu_graph, fn0)
    tensorlike_test(xpu_graph, fn1)
