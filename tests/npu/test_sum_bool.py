import torch
import torch_npu

import inductor_npu
import xpu_graph
from xpu_graph.test_utils import is_similar
from xpu_graph.config import OptLevel

def run(x):
    return torch.ops.aten.sum.dim_IntList(x, None)

def run_test():
    xpu_graph_backend = xpu_graph.npu_compiler(opt_level=OptLevel.level2)

    run_compiled = torch.compile(run, backend=xpu_graph_backend, dynamic=False)

    x = torch.empty((86, 128)).bool().npu()

    out1 = run(x)
    out_compiled1 = run_compiled(x)
    assert is_similar(out1.float(), out_compiled1.float())

    y = torch.empty((86, 256)).bool().npu()
    out2 = run(y)
    out_compiled2 = run_compiled(y)
    assert is_similar(out2.float(), out_compiled2.float())


if __name__ == "__main__":
    run_test()