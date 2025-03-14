from typing import Optional

import torch
from torch import nn, fx
from typing import Callable
from xpu_graph.config import OptLevel

from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from xpu_graph.utils import logger
from ..utils.check_ops import (
    check_gather_op,
    get_shape,
)

def find_gather_nodes(graph_module):
    candi_nodes = []
    for node in graph_module.graph.nodes:
        if not check_gather_op(node):
            continue
        if node.args[1] == 0:
            continue # not support gather dim 0
        candi_nodes.append(node)
    return candi_nodes




# This pattern is for gather ops whose gather dim's indices is generated from 'iota'
# and gather indices doesn't changed during the process, which means the index is continuous, 
# so that we can shortcut gather op by load&store op.
# TODO: currently, we only support such senario in qianchuan model,
# in which gather ops always generated from 'iota', so that this pattern just check gatherop itself.
class SingleContinuousGather(Pattern):
    _pattern_group = PatternGroup.GROUP1

    def __init__(self, target_mod: torch.nn.Module):
        self.target_mod = target_mod

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        changed = False
        candi_nodes = find_gather_nodes(graph_module)
        graph_module.add_submodule("single_continuous_gather_op", self.target_mod())
        for i,node in enumerate(candi_nodes):
            # grouped by output_len
            src,dim,idx = node.args
            prefix_len = get_shape(idx)[dim]
            
            with graph_module.graph.inserting_before(node):
                gather_node = graph_module.graph.call_module(
                    "single_continuous_gather_op", args=(src, dim, prefix_len)
                )

            node.replace_all_uses_with(gather_node)
            graph_module.graph.erase_node(node)
            changed = True

        graph_module.graph.lint()
        graph_module.recompile()

        return changed
