import torch
from torch import nn, fx
# import torch_mlu
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from typing import Callable

def custom_getitem(tensor_list, index):
    return tensor_list[index]


def find_last_placeholder(graph_module: fx.GraphModule) -> fx.Node:
    last_placeholder = None
    for node in graph_module.graph.nodes:
        if node.op == "placeholder":
            last_placeholder = node
    return last_placeholder

def check_copy_replace(node: fx.Node) -> bool:
    if node.op != 'call_function':
        return False
    if node.target != torch.ops.aten._to_copy.default:
        return False
    #import pdb;pdb.set_trace()
    if node.args[0].op != "placeholder":
        return False

    #consider cast only
    if len(node.kwargs) > 1:
        return False
    if "dtype" not in node.kwargs:
        return False
    return True

class FusedCopyReplacement(nn.Module):
    def forward(self, candidates, dtypes):
        new_candidates = []
        for i in range(len(candidates)):
            new_candidates.append(candidates[i].to(dtypes[i]))
        return tuple(new_candidates)

# one src node slice to multi dst nodes
class FusedCopyCast(Pattern):

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        graph_module.add_submodule("fused_to_copy", FusedCopyReplacement())

        last_placeholder = find_last_placeholder(graph_module)
        if last_placeholder is None:
            return changed 

        candidates = [node for node in graph_module.graph.nodes if check_copy_replace(node)]
        print(graph_module.graph)
        print("JYJ", candidates)
        print(candidates)
        if len(candidates) < 5:
            return changed

        dtypes = [node.kwargs["dtype"] for node in candidates]
        candidate_inputs = [node.args[0] for node in candidates]

        with graph_module.graph.inserting_before(candidates[0]):
            new_node = graph_module.graph.call_module(
                "fused_to_copy",
                args=(candidate_inputs, dtypes),
            )
        for idx, n in enumerate(candidates):
            # output: [src_node[0], slice_len]
            with graph_module.graph.inserting_before(n):
                with graph_module.graph.inserting_after(new_node):
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
