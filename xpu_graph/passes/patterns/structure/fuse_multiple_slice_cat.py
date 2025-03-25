import torch
from torch import nn, fx
from xpu_graph.passes.patterns.pattern import Pattern
from typing import Callable


def fuse_multiple_cat(graph_module: fx.GraphModule):
    changed = False
    cat_pattern_all = {}
    #import pdb;pdb.set_trace()
    for node in graph_module.graph.nodes:
        if node.target == "fuse_slice_cat":
            if node.args[0] not in cat_pattern_all:
                cat_pattern_all[node.args[0]] = [(node, node.args[1])]
            else:
                cat_pattern_all[node.args[0]].append((node, node.args[1]))
    for src_node in cat_pattern_all:
        if len(cat_pattern_all[src_node]) == 1:
            continue
        ori_nodes = []
        cat_input = []
        slice_offsets = []
        offset = 0
        for v in cat_pattern_all[src_node]:
            ori_nodes.append(v[0])
            cat_input += v[1]
            slice_len = 0
            for p in v[1]:
                slice_len += p[1] - p[0]
            slice_offsets.append((offset, offset + slice_len))
            offset += slice_len
        #import pdb;pdb.set_trace()
        with graph_module.graph.inserting_before(ori_nodes[0]):
            new_nodes = graph_module.graph.call_module(
                "fuse_slice_cat",
                args=(src_node, cat_input),
            )
        #import pdb;pdb.set_trace()
        for idx, ori_node in enumerate(ori_nodes):
            with graph_module.graph.inserting_before(ori_node):
                new_node = graph_module.graph.create_node(
                    op="call_function",
                    name=f"slice_node_{ori_node.name}_{idx}",
                    target=torch.ops.aten.slice.Tensor,
                    args=(new_nodes, 1, slice_offsets[idx][0], slice_offsets[idx][1]),
                    kwargs=None,
                )
                
                ori_node.replace_all_uses_with(new_node)
        changed = True
    #import pdb;pdb.set_trace()
    return changed


class FusedMultipleSliceCat(Pattern):
    def __init__(self, target_mod: torch.nn.Module):
        self.target_mod = target_mod

    def process(self, graph_module: fx.GraphModule):
        #import pdb;pdb.set_trace()
        changed = False
        graph_module.add_submodule(
            "fuse_slice_cat",
            self.target_mod(),
        )

        # merge multiple "fuse_slice_cat" with the same src_node to one "fuse_slice_cat"
        changed = changed | fuse_multiple_cat(graph_module)
        # graph_module.graph.lint()
        # graph_module.recompile()
        return changed
