import typing

import torch
import torch.fx as fx
from torch.utils._pytree import tree_flatten

from xpu_graph.passes.optimizer import Optimizer

class Cse(Optimizer):
    def process(self, gm: fx.GraphModule):
        from torch._functorch.compile_utils import fx_graph_cse
        cse_graph = fx_graph_cse(gm.graph)

        changed = len(cse_graph.nodes) != len(gm.graph.nodes)
        gm.graph = cse_graph

        return changed