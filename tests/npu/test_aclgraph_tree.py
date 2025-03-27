import torch
import torch_npu
import inductor_npu
import xpu_graph
from xpu_graph.test_utils import is_similar
from xpu_graph.config import OptLevel

class GraphTest1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden_states):
        tmp1 = torch.nn.functional.relu(hidden_states)
        tmp2 = tmp1.mul(2)
        tmp3 = tmp2 + 12.0
        return tmp3

gt1 = GraphTest1().npu()

def aclgraph_tree_test(xpu_graph, func):
    with torch.no_grad():
        inp = torch.randn(4, 32).npu()
        torch._inductor.config.triton.cudagraphs = True
        compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
        for i in range(5):
            compiled(inp)
        out = compiled(inp)
        out1 = func(inp)
        assert is_similar(out1.float(), out.float())

if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.npu_compiler(opt_level=OptLevel.level2)
    aclgraph_tree_test(xpu_graph_backend, gt1)
