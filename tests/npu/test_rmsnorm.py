import torch
import torch_npu
import inductor_npu
import xpu_graph
from xpu_graph.test_utils import is_similar
from xpu_graph.config import OptLevel

class RMSNorm1(torch.nn.Module):
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

fn0 = RMSNorm1(hidden_size=(10,)).npu()

def rmsnorm_test(xpu_graph, func):
    with torch.no_grad():
        a = torch.randn(1, 10).npu()
        compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
        norm = compiled(a)
        norm1 = func(a)
        assert is_similar(norm1.float(), norm.float())

if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.npu_compiler(opt_level=OptLevel.level2)
    rmsnorm_test(xpu_graph_backend, fn0)
