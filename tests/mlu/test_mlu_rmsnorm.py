import pytest
import torch
import torch_mlu
import xpu_graph
from xpu_graph.test_utils import is_similar


class Qwen2RMSNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


rmsnorm = Qwen2RMSNorm(hidden_size=(10,)).mlu()


def rmsnorm_test(xpu_graph, func):
    a = torch.randn(1, 10).mlu()
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    res = compiled(a)
    res1 = func(a)
    assert is_similar(res1.float(), res.float())


class TestRMSNorm:
    def setup_class(self):
        config = xpu_graph.config.XpuGraphConfig()
        config.target = xpu_graph.config.Target.mlu
        config.vendor_compiler = {"mode": "reduce-overhead"}
        self.xpu_graph = xpu_graph.compiler.XpuGraph(config)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            rmsnorm,
        ],
    )
    def test_rmsnorm_patterns(self, pattern_func):
        rmsnorm_test(self.xpu_graph, pattern_func)
