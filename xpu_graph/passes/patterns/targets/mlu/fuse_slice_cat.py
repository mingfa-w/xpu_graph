import torch
from torch import nn, fx
import torch_mlu
from xpu_graph.passes.patterns.pattern import Pattern
from ...utils.check_ops import (
    check_cat_op,
    check_slice_op,
    check_stack_op,
    check_meta_2d,
)

from .triton_kernel.triton_fused_slice_cat import (
    mlu_triton_fused_slice_cat,
)

from .triton_kernel.triton_fused_slice import (
    mlu_triton_fused_slice_low,
)

MAX_INT64 = 9223372036854775807


def apply_slice_operation(input_tensor, indices):
    rows, _ = input_tensor.shape
    indices_tensor = torch.tensor(
        indices, dtype=torch.int32, device=input_tensor.device
    )
    return mlu_triton_fused_slice_cat(
        input_tensor,
        indices_tensor,
        rows,
        len(indices),
        input_tensor.stride(0),
        16384,  # blocksize
    )


class FuseSliceCatSameInputReplacement(nn.Module):
    def forward(self, input_tensor, slices):
        if len(input_tensor.shape) != 2:
            raise NotImplementedError("input must be 2d")
        indices = [i for start, end in slices for i in range(start, end)]
        return apply_slice_operation(input_tensor, indices)


class CatReplacement(nn.Module):
    def forward(self, input_tensor_list, axis=-1):
        return torch.cat(input_tensor_list, axis=axis)


class MergeCatReplacement(nn.Module):
    def forward(self, input_tensor_list, axis=0):
        return torch.cat(
            [
                (
                    input_tensor
                    if len(input_tensor.shape) == 3
                    else input_tensor.unsqueeze(0)
                )
                for input_tensor in input_tensor_list
            ],
            axis=0,
        )


class ExpandTransReplacement(nn.Module):
    def forward(self, input_tensor, dim):
        return input_tensor.reshape(input_tensor.shape[0], dim, -1).transpose(0, 1)


def validate_slice_operation(n_list):
    if len(n_list) < 2:
        return False, None, None
    slice_input = []
    slice_param = []
    slice_axis = []
    for n in n_list:
        slice_input.append(n.args[0])
        slice_axis.append(n.args[1])
        right = n.args[3]
        if right == MAX_INT64:
            right = slice_input[0].meta["tensor_meta"].shape[-1]
        elif right < 0:
            right = slice_input[0].meta["tensor_meta"].shape[-1] - (-right)
        slice_param.append((n.args[2], right))
    if slice_input.count(slice_input[0]) != len(slice_input):
        return False, None, None
    if slice_axis.count(1) != len(slice_axis):
        return False, None, None
    return True, slice_input[0], slice_param


def extract_slice_info(nodes):
    slice_input = []
    slice_axis = []
    slice_param = []

    for node in nodes:
        if not check_slice_op(node):
            return False, [], [], []
        slice_input.append(node.args[0])
        slice_axis.append(node.args[1])
        slice_param.append((node.args[2], node.args[3]))

    return True, slice_input, slice_axis, slice_param


def insert_fuse_slice(gm, node, src_node, slice_param):
    with gm.graph.inserting_before(node):
        slice_node = gm.graph.call_module(
            "mlu_triton_fuse_slice_cat_same_input_module",
            args=(src_node, slice_param),
        )
    return slice_node


def fuse_mixed_ops_and_catstack(gm: fx.GraphModule):
    changed = False
    for node in reversed(gm.graph.nodes):
        is_cat = check_cat_op(node)
        if (not check_stack_op(node)) and (not is_cat):
            continue
        if is_cat:
            if not check_meta_2d(node):
                continue
            axis = node.args[1]
            if axis == 0:
                continue
        changed1 = False
        cat_input_list = []
        n_list = []
        for n in node.args[0]:
            if check_slice_op(n) and check_meta_2d(n):
                n_list.append(n)
            else:
                is_slice, src_node, slice_param = validate_slice_operation(n_list)
                if is_slice:
                    slice_node = insert_fuse_slice(gm, node, src_node, slice_param)
                    if is_cat:
                        cat_input_list.append(slice_node)
                    else:
                        with gm.graph.inserting_before(node):
                            stack_node = gm.graph.call_module(
                                "expand_transpose_module",
                                args=(slice_node, len(slice_param)),
                            )
                        cat_input_list.append(stack_node)
                    changed1 = True
                else:
                    cat_input_list += n_list
                cat_input_list.append(n)
                n_list = []
        is_slice, src_node, slice_param = validate_slice_operation(n_list)
        if is_slice:
            slice_node = insert_fuse_slice(gm, node, src_node, slice_param)
            if is_cat:
                cat_input_list.append(slice_node)
            else:
                with gm.graph.inserting_before(node):
                    stack_node = gm.graph.call_module(
                        "expand_transpose_module",
                        args=(slice_node, len(slice_param)),
                    )
                cat_input_list.append(stack_node)
            changed1 = True
        else:
            cat_input_list += n_list

        if changed1 == True:
            if is_cat:
                with gm.graph.inserting_before(node):
                    cat_node = gm.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.cat.default,
                        args=(cat_input_list, -1),
                        name=node.name + "_replacement",
                    )
            else:
                with gm.graph.inserting_before(node):
                    cat_node = gm.graph.call_module(
                        "mlu_triton_fused_stack", args=(cat_input_list, -1)
                    )
            node.replace_all_uses_with(cat_node)
            slice_nodes = node.args[0]
            for slice_node in slice_nodes:
                if len(slice_node.users) == 0:
                    gm.graph.erase_node(slice_node)

            gm.graph.erase_node(node)
            changed = True

    return changed


def fuse_multiple_cat(gm: fx.GraphModule):
    changed = False
    cat_pattern_all = {}
    for node in gm.graph.nodes:
        if node.target == "mlu_triton_fuse_slice_cat_same_input_module":
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
        with gm.graph.inserting_before(ori_nodes[0]):
            new_nodes = gm.graph.call_module(
                "mlu_triton_fuse_slice_cat_same_input_module",
                args=(src_node, cat_input),
            )
        for idx, ori_node in enumerate(ori_nodes):
            with gm.graph.inserting_before(ori_node):
                new_node = gm.graph.create_node(
                    op="call_function",
                    name=f"slice_node_{ori_node.name}_{idx}",
                    target=torch.ops.aten.slice.Tensor,
                    args=(new_nodes, 1, slice_offsets[idx][0], slice_offsets[idx][1]),
                    kwargs=None,
                )
                ori_node.replace_all_uses_with(new_node)
    return changed


class FuseSliceTogetherReplacement(nn.Module):
    def forward(self, input_tensor, slices_index, slice_len):
        if len(input_tensor.shape) != 2:
            raise NotImplementedError("input must be 2d")
        if slice_len > input_tensor.shape[-1]:
            raise NotImplementedError(
                f"inputshape {input_tensor.shape} don't support slice_len:{slice_len}"
            )
        slices_index = torch.tensor(
            slices_index, dtype=torch.int32, device=input_tensor.device
        )
        output = mlu_triton_fused_slice_low(
            input_tensor,
            slices_index,
            slice_len,
            input_tensor.shape[0],
            input_tensor.stride(0),
        )
        return output


def custom_getitem(tensor_list, index):
    return tensor_list[index]


def fuse_slice_together(gm: fx.GraphModule):
    changed = False
    gm.add_submodule("mlu_triton_fused_slice_together", FuseSliceTogetherReplacement())

    def divide_nodes_in_slice_len(nodes):
        divide_nodes = {}
        for n in nodes:
            slice_len = n.args[3] - n.args[2]
            if slice_len not in divide_nodes:
                divide_nodes[slice_len] = []
            divide_nodes[slice_len].append((n, n.args[2]))
        return divide_nodes

    candi_nodes = {}
    for node in gm.graph.nodes:
        if not check_slice_op(node):
            continue
        if len(node.users) == 1:
            if next(iter(node.users)).target == "output":
                continue
        if node.args[0] not in candi_nodes:
            candi_nodes[node.args[0]] = []
        candi_nodes[node.args[0]].append(node)

    """
    result_node = list(gm.graph.nodes)[-1]
    merge_nodes = {}
    for n in result_node.args[0]:
        if not check_slice_op(n):
            continue
        if len(n.users) != 1: 
            continue
        if n.args[1] != 1:
            continue
        if n.args[0] not in merge_nodes:
            merge_nodes[n.args[0]] = []
        merge_nodes[n.args[0]].append(n)
    """
    for src_node, nodes in candi_nodes.items():
        divide_nodes = divide_nodes_in_slice_len(nodes)
        for slice_len, nodes2 in divide_nodes.items():
            if len(nodes2) < 3:
                continue
            start_indices = [n[1] for n in nodes2]
            replace_n = [n[0] for n in nodes2]
            with gm.graph.inserting_before(replace_n[0]):
                new_node = gm.graph.call_module(
                    "mlu_triton_fused_slice_together",
                    args=(src_node, start_indices, slice_len),
                )
            for idx, n in enumerate(replace_n):
                with gm.graph.inserting_before(n):
                    new_n = gm.graph.create_node(
                        op="call_function",
                        target=custom_getitem,
                        args=(new_node, idx),
                        name=f"getitem_node_{new_node.name}_{n.name}",
                    )
                n.replace_all_uses_with(new_n)
                gm.graph.erase_node(n)
            changed = True
    return changed


class FusedSlice(Pattern):
    def process(self, gm: fx.GraphModule):
        changed = False
        gm.add_submodule(
            "mlu_triton_fuse_slice_cat_same_input_module",
            FuseSliceCatSameInputReplacement(),
        )
        gm.add_submodule("expand_transpose_module", ExpandTransReplacement())
        gm.add_submodule("mlu_triton_fused_cat", CatReplacement())
        gm.add_submodule("mlu_triton_fused_stack", MergeCatReplacement())

        # slice & cat, the inputs of cat are mixed with slice and other ops.
        changed = changed | fuse_mixed_ops_and_catstack(gm)

        # merge multiple cat ops into one
        changed = changed | fuse_multiple_cat(gm)

        changed = changed | fuse_slice_together(gm)

        gm.graph.lint()
        gm.recompile()
        return changed
