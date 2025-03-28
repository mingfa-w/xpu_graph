import pytest
import torch
import xpu_graph
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache


def fn0(a, b):
    return torch.sum(torch.stack([a, b]), dim=0)

def reduce_test(xpu_graph, func):
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    a = torch.randn(128,8)
    b = torch.randn(128,8)
    res = func(a,b)
    res1 = compiled(a,b)
    assert torch.equal(res.float(), res1.float())


class TestReduce:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
        ],
    )
    def test_reduce_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph):
            reduce_test(self.xpu_graph, pattern_func)
        assert "Pattern.FoldReduce changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False, debug=True)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    reduce_test(xpu_graph, fn0)
