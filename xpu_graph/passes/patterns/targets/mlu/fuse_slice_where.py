import torch
import operator
from torch import nn, fx
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from xpu_graph.config import OptLevel
from ...utils.check_ops import (
    check_where_op,
    check_slice_op,
    check_zeros_op,
)
from .triton_kernel.fused_slice_where import fuse_slice_where

class FusedSliceWhereReplacement(nn.Module):
    def forward(
        self,
        where_input,
        slice_input,
        slice_len,
        start_indices,
    ):
        slice_num = int(len(start_indices))
        start_indices_tensor = torch.tensor(start_indices, dtype=torch.int32, device=where_input.device)
        output = fuse_slice_where(where_input, slice_input, start_indices_tensor, slice_len, slice_num)
        return output.view(slice_num, slice_input.shape[0], slice_len)

def custom_getitem(tensor_list, index):
    return tensor_list[index]

def divide_slice_where_nodes(nodes):
    divide_nodes = {}
    for n in nodes:
        slice_node = n.args[2]
        slice_len = slice_node.args[3] - slice_node.args[2]
        if slice_len not in divide_nodes:
            divide_nodes[slice_len] = []
        divide_nodes[slice_len].append((slice_node, n, slice_node.args[2]))
    return divide_nodes

def find_slice_where_nodes(graph_module):
    candi_nodes = {}
    for node in graph_module.graph.nodes:
        if (
            not check_where_op(node)
            or not check_slice_op(node.args[2])
            or not check_zeros_op(node.args[1])
        ):
            continue

        slice_input = node.args[2].args[0]
        slice_dim = node.args[2].args[1]
        if not (
            hasattr(slice_input, "meta")
            and "tensor_meta" in slice_input.meta
            and len(slice_input.meta["tensor_meta"].shape) == 2
            and slice_dim in [1, -1]
        ):
            continue

        if len(node.users) == 1:
            if next(iter(node.users)).target == "output":
                continue

        key = (node.args[0], slice_input)
        if key not in candi_nodes:
            candi_nodes[key] = []
        candi_nodes[key].append(node)
    return candi_nodes

class FusedSliceWhere(Pattern):
    _opt_level = OptLevel.level2
    _pattern_group = PatternGroup.GROUP1

    def process(self, graph_module: fx.GraphModule) -> bool:
        is_modified = False
        graph_module.add_submodule("mlu_triton_slice_where_replacement", FusedSliceWhereReplacement())
        candi_nodes = find_slice_where_nodes(graph_module)

        for src_nodes, nodes in candi_nodes.items():
            divide_nodes = divide_slice_where_nodes(nodes)

            for slice_len, nodes_ in divide_nodes.items():
                if len(nodes_) < 2:
                    continue
                start_indices = [n[2] for n in nodes_]
                where_ = [n[1] for n in nodes_]
                slice_ = [n[0] for n in nodes_]
                with graph_module.graph.inserting_before(where_[0]):
                    new_node = graph_module.graph.call_module(
                        "mlu_triton_slice_where_replacement",
                        args=(src_nodes[0], src_nodes[1], slice_len, start_indices),
                    )
                for idx, n in enumerate(where_):
                    with graph_module.graph.inserting_before(n):
                        new_n = graph_module.graph.create_node(
                            op="call_function",
                            target=operator.getitem,
                            args=(new_node, idx),
                        )
                    n.replace_all_uses_with(new_n)
                    graph_module.graph.erase_node(n)
                is_modified = True

        graph_module.graph.lint()
        graph_module.recompile()
        return is_modified
