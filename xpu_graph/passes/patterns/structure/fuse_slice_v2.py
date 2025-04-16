import torch
from torch import nn, fx
# import torch_mlu
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from typing import Callable
from ..utils.check_ops import (
    check_slice_op,
    check_stack_op,
)


def custom_getitem(tensor_list, index):
    return tensor_list[index]


def divide_nodes_in_slice_len(nodes):
    """
    Groups nodes based on the length of their slice operation.

    Args:
        nodes (list): A list of nodes, where each node represents a slice operation.

    Returns:
        dict: A dictionary where keys are slice lengths (slice_end - slice_start),
              and values are lists of tuples, each containing a node and its slice start index.
    """
    divide_nodes = {}
    for n in nodes:
        slice_len = n.args[3] - n.args[2]
        if slice_len not in divide_nodes:
            divide_nodes[slice_len] = []
        divide_nodes[slice_len].append((n, n.args[2]))
    return divide_nodes


def find_slice_nodes(graph_module):
    """
    Identifies and groups slice operation nodes in a given graph module.

    Args:
        graph_module: A graph module containing computation nodes.

    Returns:
        dict: A dictionary where keys are parent nodes, and values are lists of
              associated slice operation nodes.
    """
    candi_nodes = {}
    for node in graph_module.graph.nodes:
        if not check_slice_op(node):
            continue
        # Skip slice nodes that are directly connected to the output node.
        '''
        if len(node.users) == 1:
            if next(iter(node.users)).target == "output":
                continue
        '''
        if node.args[0] not in candi_nodes:
            candi_nodes[node.args[0]] = []
        candi_nodes[node.args[0]].append(node)
    return candi_nodes


# one src node slice to multi dst nodes
class FusedSliceV2(Pattern):
    _pattern_group = PatternGroup.GROUP2

    def __init__(self, target_mod: torch.nn.Module, *super_args):
        super().__init__(*super_args)
        self.target_mod = target_mod

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        graph_module.add_submodule("fused_slice_v2", self.target_mod())
        candi_nodes = find_slice_nodes(graph_module)

        for src_node, nodes in candi_nodes.items():
                if len(nodes) < 2:
                    continue
                start_indices = [n.args[2] for n in nodes]
                slice_lens = [n.args[3] - n.args[2] for n in nodes]
                # output: [num, src_node[0], slice_len]
                with graph_module.graph.inserting_before(nodes[0]):
                    new_node = graph_module.graph.call_module(
                        "fused_slice_v2",
                        args=(src_node, start_indices, slice_lens),
                    )
                for idx, n in enumerate(nodes):
                    # output: [src_node[0], slice_len]
                    with graph_module.graph.inserting_before(n):
                        new_n = graph_module.graph.create_node(
                            op="call_function",
                            target=custom_getitem,
                            args=(new_node, idx),
                            name=f"getitem_node_{new_node.name}_{n.name}",
                        )
                    n.replace_all_uses_with(new_n)
                    graph_module.graph.erase_node(n)
                changed = True
        graph_module.graph.lint()
        graph_module.recompile()
        return changed
