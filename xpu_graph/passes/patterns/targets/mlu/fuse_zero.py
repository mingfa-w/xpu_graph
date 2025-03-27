import torch
from torch import nn, fx
# import torch_mlu
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from typing import Callable
from xpu_graph.fx_utils import FxStage

def get_node_meta(node):
    node_meta = {}
    node_meta["shape"] = node.args[0]
    node_meta["dtype"] = node.kwargs.get("dtype") 
    node_meta["device"] = node.kwargs.get("device") 
    #mlu device don't need pin memory
    return node_meta
    

def custom_getitem(tensor_list, index):
    return tensor_list[index]

#%zeros_default : [num_users=1] = call_function[target=torch.ops.aten.zeros.default](args = ([2048, 1],), kwargs = {dtype: torch.float32, device: mlu:0, pin_memory: False})
class FusedZero(Pattern):
    _pattern_group = PatternGroup.GROUP1
    _stages = [FxStage.inference, FxStage.pregrad]

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False

        candidates = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function" and node.target == torch.ops.aten.zeros.default 
        ]
        print(graph_module.graph)
        if len(candidates) < 2:
            return False
        #import pdb;pdb.set_trace()
        cand_dict = {}
        for node in candidates:
            node_meta = get_node_meta(node)
            if node_meta["device"].type != torch.device('mlu').type:
                continue
            node_meta_hash = tuple(sorted(node_meta.items()))
            if node_meta_hash not in cand_dict:
                cand_dict[node_meta_hash] = [node]
            else:
                cand_dict[node_meta_hash].append(node)
        for key, value in cand_dict.items(): 
            if len(value) < 2:
                continue
            shape = []
            dtype = torch.float32
            for sub_key in key:
                if "shape" == sub_key[0]:
                    shape = sub_key[1]
                if "dtype" == sub_key[0]:
                    dtype = sub_key[1]
            new_node_name = str([v.name for v in value])
            shape = [len(value)] + shape
            with graph_module.graph.inserting_before(value[0]):
                zero_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.zeros,
                    args=(shape,),
                    kwargs={'dtype': dtype, 'device': 'mlu'},
                    name=new_node_name
                )
            for idx in range(len(value)):
                n = value[idx]
                with graph_module.graph.inserting_before(n):
                    new_n = graph_module.graph.create_node(
                        op="call_function",
                        target=custom_getitem,
                        args=(zero_node, idx),
                        name=f"getitem_node_{zero_node.name}_{n.name}",
                    )
                n.replace_all_uses_with(new_n)
                graph_module.graph.erase_node(n)
        graph_module.graph.lint()
        graph_module.recompile()
        return changed


'''
from torch.fx import GraphModule

def dedup_scalar_zeros(gm: GraphModule):
    zero_const = torch.tensor(0.0, dtype=torch.float32, device='mlu')
    found_zeros = []
    for node in gm.graph.nodes:
        if node.op == "call_function" and node.target == torch.ops.aten.zeros.default:
            if node.args == ([],) and node.kwargs.get("dtype") == torch.float32:
                found_zeros.append(node)
    # 用第一个替代所有
    if found_zeros:
        with gm.graph.inserting_before(found_zeros[0]):
            const_node = gm.graph.create_node("get_attr", "shared_zero_scalar")
        for n in found_zeros:
            n.replace_all_uses_with(const_node)
            gm.graph.erase_node(n)
        gm.register_buffer("shared_zero_scalar", zero_const)
    gm.recompile()
    return gm

'''
