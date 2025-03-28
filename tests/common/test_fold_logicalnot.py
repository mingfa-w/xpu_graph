import pytest
import torch
import xpu_graph
from xpu_graph.test_utils import need_xpu_graph_logs, skip_xpu_graph_cache


def fn0():
    x = torch.ones((32, 32), dtype=torch.float32, device="cpu")
    y = torch.ones((32, 32), dtype=torch.float32, device="cpu")
    return torch.logical_not(x == y)


def fn1():
    x = torch.ones((32, 32), dtype=torch.float32, device="cpu")
    return torch.logical_not(x)


def logicalnot_test(xpu_graph, func):
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    res = func()
    res1 = compiled()
    assert torch.equal(res.cpu().float(), res1.cpu().float())


class TestView:
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
    def test_logicalnot_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph):
            logicalnot_test(self.xpu_graph, pattern_func)
            assert "Pattern.FoldLogicalNot changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False, debug=True)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    logicalnot_test(xpu_graph, fn0)
    logicalnot_test(xpu_graph, fn1)
