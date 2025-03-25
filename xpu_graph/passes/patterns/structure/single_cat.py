from typing import Optional

import torch
from torch import nn, fx
from typing import Callable
from xpu_graph.config import OptLevel

from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from xpu_graph.utils import logger
from ..utils.check_ops import (
    check_cat_op,
)

def is_2d_tensor_meet_ub(node_inpus, dtype_bytes):
    sum_size = 0
    for inp in node_inpus:
        # concat inp tensor memory is contiguous
        if inp.meta["tensor_meta"].memory_format != torch.contiguous_format:
            return False
        # only cancat 2d tensor in triton dsl
        shape = inp.meta["tensor_meta"].shape
        if len(shape) != 2:
            return False
        if sum_size * dtype_bytes > 163840:
        # triton load all cat tensor args into memory (160 kb = 131072bytes), set by developer
            return False
        sum_size += (shape[0] * shape[1])
    return True


def find_single_cat_nodes(graph_module):
    candi_nodes = []
    for node in graph_module.graph.nodes:
        is_cat, cat_axis = check_cat_op(node)
        if not is_cat:
            continue
        elif node.meta == {}:
            continue
        else:
            # only support aten.cat.default((2d tensor1,tensor2...), dim=0/1)
            inps = node.args[0]
            # 10 - 15 input tensor, set by developer
            if len(inps) < 4 or len(inps) == 6 or len(inps) > 12:
            # len 6 have nan data input
                continue
            elif cat_axis == 0:
            # only concat dim=1
                continue
            if (inps[0].meta["tensor_meta"].dtype == torch.float16):
                dtype_bytes = 2
            if is_2d_tensor_meet_ub(inps, dtype_bytes):
                candi_nodes.append(node)
    return candi_nodes


class SingleCatTwoDim(Pattern):
    _pattern_group = PatternGroup.GROUP1

    def __init__(self, target_mod: torch.nn.Module):
        self.target_mod = target_mod

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        changed = False
        candi_nodes = find_single_cat_nodes(graph_module)
        graph_module.add_submodule("single_cat_op", self.target_mod())
        for node in candi_nodes:
            # node is aten.cat
            src, dim = node.args
            with graph_module.graph.inserting_before(node):
                cat_node = graph_module.graph.call_module(
                    "single_cat_op", args=(src, dim)
                )

            node.replace_all_uses_with(cat_node)
            graph_module.graph.erase_node(node)
            changed = True

        graph_module.graph.lint()
        graph_module.recompile()

        return changed