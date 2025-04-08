import torch
import pytest
from xpu_graph import XpuGraph, XpuGraphConfig
from xpu_graph.test_utils import (
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
    is_similar,
)


class CanConstantFolding1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.rand(128, 128), requires_grad=False)

    @torch.no_grad()
    def forward(self, x):
        weight = torch.relu(self.weight)
        return torch.matmul(x, weight)


class CanConstantFolding2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_0 = torch.nn.Parameter(torch.rand(128, 64), requires_grad=False)
        self.weight_1 = torch.nn.Parameter(torch.rand(128, 64), requires_grad=False)

    @torch.no_grad()
    def forward(self, x):
        weight = torch.cat([self.weight_0, self.weight_1], dim=1)
        return torch.matmul(x, weight)


class CanConstantFolding3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight_0 = torch.nn.Parameter(torch.ones(128, 64), requires_grad=False)
        self.weight_1 = torch.nn.Parameter(torch.ones(128, 64), requires_grad=False)

    @torch.no_grad()
    def forward(self, x):
        return torch.matmul(x, self.weight_0) + torch.matmul(x, self.weight_1)


class TestConstantFolding:
    def setup_class(self):
        config = XpuGraphConfig(freeze=True, is_training=False, constant_folding=True)
        self.xpu_graph = XpuGraph(config)

    @pytest.mark.parametrize(
        "testcase",
        [
            CanConstantFolding1,
            CanConstantFolding2,
            CanConstantFolding3,
        ],
    )
    def test_constant_folding(self, caplog, testcase):
        mod = testcase()
        input = torch.rand(128, 128)
        expect = mod(input)
        with need_xpu_graph_logs():
            compiled_mod = torch.compile(mod, backend=self.xpu_graph, dynamic=False)
            result = compiled_mod(input)

        assert (
            is_similar(result, expect)
            and "Optimizer.ConstantFolding changed graph" in caplog.text
        )


if __name__ == "__main__":

    class CanConstantFolding1(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight_0 = torch.nn.Parameter(torch.rand(128, 64), requires_grad=False)
            self.weight_1 = torch.nn.Parameter(torch.rand(128, 64), requires_grad=False)

        @torch.no_grad()
        def forward(self, x):
            weight = torch.cat([self.weight_0, self.weight_1], dim=1)
            return torch.matmul(x, weight)

    mod = CanConstantFolding1()
    xpu_graph_backend = XpuGraph(XpuGraphConfig(freeze=True, is_training=False))
    with need_xpu_graph_logs(), skip_xpu_graph_cache(xpu_graph_backend):
        compiled_mod = torch.compile(mod, backend=xpu_graph_backend, dynamic=False)
        res = compiled_mod(torch.ones(128, 128))
        expect = mod(torch.ones(128, 128))

    assert (
        is_similar(res, expect)
        # and "Optimizer.ConstantFolding changed graph" in caplog.text
    )

    class CanConstantFolding2(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight_0 = torch.nn.Parameter(torch.ones(128, 64), requires_grad=False)
            self.weight_1 = torch.nn.Parameter(torch.ones(128, 64), requires_grad=False)
            self.weight_2 = torch.nn.Parameter(
                torch.ones(128, 128), requires_grad=False
            )

        @torch.no_grad()
        def forward(self, x):
            # weight = torch.cat([self.weight_0, self.weight_1], dim=1)
            return torch.matmul(x, self.weight_0) + torch.matmul(x, self.weight_1)

    mod = CanConstantFolding2()
    xpu_graph_backend = XpuGraph(XpuGraphConfig(freeze=True, is_training=False))
    with need_xpu_graph_logs(), skip_xpu_graph_cache(xpu_graph_backend):
        compiled_mod = torch.compile(mod, backend=xpu_graph_backend, dynamic=False)
        res = compiled_mod(torch.ones(128, 128))
        expect = mod(torch.ones(128, 128))

    assert (
        is_similar(res, expect)
        # and "Optimizer.ConstantFolding changed graph" in caplog.text
    )
