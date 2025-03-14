from typing import Optional, Tuple, Union

import torch
from torch import nn, fx
import torch_mlu
import torch_mlu_ops
from xpu_graph.utils import logger
from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from ...utils.check_ops import get_shape, check_slice_op, check_meta_2d, check_cat_op
from xpu_graph.fx_utils import trace_and_inline, FxStage
from .triton_kernel.fused_slice_cat_d import fused_slice_cat_d
TensorShape = Union[torch.Size, Tuple[int, ...]]
NodeType = fx.Node

from torch._subclasses.fake_tensor import FakeTensor
from torch._subclasses.functional_tensor import FunctionalTensor

class FusedSliceCatDReplacement(nn.Module):
    def forward(self, src_nodes, slice_params):
        print(src_nodes)
        return torch.cat(
            [
                src_nodes[i][:, slice_params[i * 2] : slice_params[i * 2 + 1]]
                for i in range(len(src_nodes))
            ],
            dim=1,
        )
    '''
    def forward(self, src_nodes, slice_params):
        output_len = 0
        start_indices = []
        slice_lens = []
        output_offsets = []
        batch_size = src_nodes[0].shape[0]
        for i in range(len(src_nodes)):
            start_indices.append(slice_params[i*2])
            slice_len = slice_params[i*2+1] - slice_params[i*2]
            slice_lens.append(slice_len)
            output_offsets.append(output_len)
            output_len += slice_len

        print("1", src_nodes)
        if isinstance(src_nodes[0], FakeTensor):
            return torch.empty(batch_size, output_len, device=src_nodes[0].device)
        print("2", src_nodes)

        max_slice_len = max(slice_lens)

        nodes_tensorptr = torch.tensor([s.data_ptr() for s in src_nodes], device = src_node.device, dtype = torch.int64)
        start_indices = torch.tensor(start_indices, dtype = torch.int32, device = src_node.device)
        slice_lens = torch.tensor(slice_lens, dtype = torch.int32, device = src_node.device)
        output_offsets = torch.tensor(output_offsets, dtype = torch.int32, device = src_node.device)

        return fused_slice_cat_d(src_nodes, start_indices, slice_lens, output_offsets, batch_size, output_len, max_slice_len)
    '''

def match_sub_list(lst):
    max_len = 0
    current_len = 0
    start_index = -1

    best_start = -1
    best_end = -1

    for i, val in enumerate(lst):
        if check_slice_op(val) and check_meta_2d(val) and (val.args[1] == 1):
            if current_len == 0:
                start_index = i
            current_len += 1
            if current_len > max_len:
                max_len = current_len
                best_start = start_index
                best_end = i
        else:
            current_len = 0
    return best_start, best_end

def insert_fuse_slice(graph_module, node, src_nodes, slice_param):
    with graph_module.graph.inserting_before(node):
        slice_cat_node = graph_module.graph.call_module(
            "mlu_fuse_slice_cat_d",
            args=(src_nodes, slice_param),
        )
    return slice_cat_node

class FusedSliceCatD(Pattern):
    _opt_level = OptLevel.level1
    #_pattern_group = PatternGroup.GROUP1
    #_stages = [FxStage.pregrad, FxStage.forward, FxStage.inference]
    def process(self, graph_module: fx.GraphModule) -> bool:

        graph_module.add_submodule(
            "mlu_fuse_slice_cat_d", FusedSliceCatDReplacement()
        )
        is_modified = False
        for node in reversed(graph_module.graph.nodes):
            is_cat, cat_axis = check_cat_op(node)
            if not is_cat:
                continue
            if not check_meta_2d(node):
                continue
            # 1 or -1
            if cat_axis == 0:
                continue
            ori_cat_input = node.args[0]
            start, end = match_sub_list(ori_cat_input)
            if end - start < 2:
                continue
            n_list = ori_cat_input[start : end + 1]
            src_nodes = []
            slice_param = []
            for n in n_list:
                src_nodes.append(n.args[0])
                slice_param += [n.args[2], n.args[3]]
            slice_cat_node = insert_fuse_slice(
                graph_module, node, src_nodes, slice_param
            )
            new_cat_input = ori_cat_input[:start]
            new_cat_input.append(slice_cat_node)
            new_cat_input += ori_cat_input[end + 1 :]
            with graph_module.graph.inserting_before(node):
                cat_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.cat.default,
                    args=(new_cat_input, -1),
                    name=node.name + "_replacement",
                )
            node.replace_all_uses_with(cat_node)
            graph_module.graph.erase_node(node)
            for slice_node in ori_cat_input:
                if len(slice_node.users) == 0:
                    graph_module.graph.erase_node(slice_node)
            is_modified = True

        return is_modified
