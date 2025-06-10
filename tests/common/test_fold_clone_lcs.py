import pytest
import torch
import xpu_graph
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs



def cat_lcs_test(xpu_graph):
    def fn0(x1, x2, x3, x4, x5):
        cat1 = torch.cat([x1, x2, x3, x5], dim=0)
        cat2 = torch.cat([x1, x2, x3, x4], dim=0)
        cat3 = torch.cat([x1, x2, x3, x4, x5], dim=0)
        return cat1, cat2, cat3


    x1 = torch.randn(10, 10)
    x2 = torch.randn(10, 10)
    x3 = torch.randn(10, 10)
    x4 = torch.randn(10, 10)
    x5 = torch.randn(10, 10)
    compiled = torch.compile(fn0, backend=xpu_graph, dynamic=False)
    expect = fn0(x1,x2,x3,x4,x5)
    res = compiled(x1,x2,x3,x4,x5)
    for i in range(3):
        assert is_similar(expect[i], res[i])


class TestCatLCS:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig(is_training=False)
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    def cat_lcs_test(self, caplog):
        with need_xpu_graph_logs():
            can_fold_test(self.xpu_graph)
            assert "Pattern.FoldClone changed graph" in caplog.text


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    cat_lcs_test(xpu_graph)
