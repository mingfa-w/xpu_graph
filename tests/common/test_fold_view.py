import pytest
import torch
import xpu_graph
from xpu_graph.test_utils import is_similar


def fn0(a):
    output = a.view(128, 64)
    return output

def fn1(a):
    a_view = a.view(1, -1)
    output = a_view.view(-1, 64)
    return output

def fn2(a):
    a_squeeze = a.squeeze()
    output = a_squeeze.view(-1, 1)
    return output

def fn3(a):
    a_squeeze = a.squeeze()
    output = a_squeeze.view(-1, 1)
    return a_squeeze, output

def fn4(a):
    a_squeeze = a.squeeze().view(1, -1).unsqueeze(dim=0)
    output = a_squeeze.view(-1, 1)
    return output

def view_test(xpu_graph, func):
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    a = torch.randn(128, 64)
    if func in [fn2]:
        a = torch.randn(128, 1, 64, 1)
    res = func(a)
    res1 = compiled(a)
    for i in range(len(res)):
        assert torch.equal(res[i].float(), res1[i].float())


class TestView:
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
        ],
    )
    def test_view_patterns(self, pattern_func):
        view_test(self.xpu_graph, pattern_func)


if __name__ == "__main__":
    config = xpu_graph.config.XpuGraphConfig(is_training=False)
    xpu_graph = xpu_graph.compiler.XpuGraph(config)
    view_test(xpu_graph, fn0)
    view_test(xpu_graph, fn1)
    view_test(xpu_graph, fn2)
    view_test(xpu_graph, fn3)
    view_test(xpu_graph, fn4)
