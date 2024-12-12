import torch
from torch import nn, fx
import torch_mlu
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.check_op import (
    check_cat_op,
    check_slice_op,
    check_stack_op,
)

from ..triton_kernel.triton_fused_slice_cat import (
    mlu_triton_fused_slice_cat,
)


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


class FuseSliceCatMixedInputReplacement(nn.Module):
    def forward(self, input_tensors, slices, axis=-1):
        output_tensor_list = []
        last_indices = []
        last_input_tensor = input_tensors[0]
        last_indices.extend(range(slices[0][0], slices[0][1]))
        for idx in range(1, len(input_tensors)):
            input_tensor = input_tensors[idx]
            if id(input_tensor) == id(last_input_tensor):
                last_indices.extend(range(slices[idx][0], slices[idx][1]))
            else:
                output_tensor_list.append(
                    apply_slice_operation(last_input_tensor, last_indices)
                )
                last_input_tensor = input_tensor
                last_indices = []
                last_indices.extend(range(slices[idx][0], slices[idx][1]))
        output_tensor_list.append(
            apply_slice_operation(last_input_tensor, last_indices)
        )
        return torch.cat(output_tensor_list, axis=axis)


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


class SliceReplacement(nn.Module):
    def forward(self, input_tensor, slice):
        return input_tensor[:, slice[0] : slice[1]]


def validate_slice_operation(node, slice_input, slice_axis):
    if slice_axis[0] != 1:
        return False
    if len(slice_input[0].meta["tensor_meta"].shape) != 2:
        return False
    return True


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


def _is_slice_stack(node: fx.Node):
    if not check_stack_op(node):
        return False, None, None, None

    valid, slice_input, slice_axis, slice_param = extract_slice_info(node.args[0])
    if not valid:
        return False, None, None, None

    if not validate_slice_operation(node, slice_input, slice_axis):
        return False, None, None, None

    has_same_input = slice_input.count(slice_input[0]) == len(slice_input)
    has_same_axis = slice_axis.count(slice_axis[0]) == len(slice_axis)

    return (
        (has_same_input and has_same_axis),
        slice_input[0],
        slice_axis[0],
        slice_param,
    )


def _is_slice_cat_with_same_input(node: fx.Node):
    if not check_cat_op(node) or node.args[1] not in (-1, 1):
        return False, None, None, None

    valid, slice_input, slice_axis, slice_param = extract_slice_info(node.args[0])
    if not valid:
        return False, None, None, None

    if not validate_slice_operation(node, slice_input, slice_axis):
        return False, None, None, None

    has_same_input = slice_input.count(slice_input[0]) == len(slice_input)
    has_same_axis = slice_axis.count(slice_axis[0]) == len(slice_axis)

    return (
        (has_same_input and has_same_axis),
        slice_input[0],
        slice_axis[0],
        slice_param,
    )


def _is_slice_cat_mixed_input(node: fx.Node):
    if check_cat_op(node):
        if node.args[1] != -1:
            return False, None, None, None
        slice_input = []
        slice_axis = []
        slice_param = []
        for slice_node in node.args[0]:
            if check_slice_op(slice_node) is False:
                return False, None, None, None
            slice_input.append(slice_node.args[0])
            slice_axis.append(slice_node.args[1])
            slice_param.append((slice_node.args[2], slice_node.args[3]))

        if slice_axis[0] != 1:
            return False, None, None, None
        matched = slice_axis.count(slice_axis[0]) == len(slice_axis)
        input_node = slice_input[0]
        for input_node in slice_input:
            if len(input_node.meta["tensor_meta"].shape) != 2:
                return False, None, None, None
        return matched, slice_input, slice_axis[0], slice_param

    return False, None, None, None


def fuse_slice_cat_with_same_input(gm: fx.GraphModule):
    changed = False
    for node in reversed(gm.graph.nodes):
        matched, src_node, axis, slice_param = _is_slice_cat_with_same_input(node)
        if matched:
            with gm.graph.inserting_before(node):
                slice_node = gm.graph.call_module(
                    "mlu_triton_fuse_slice_cat_same_input_module",
                    args=(src_node, slice_param),
                )
            node.replace_all_uses_with(slice_node)
            changed = True
            slice_nodes = node.args[0]
            gm.graph.erase_node(node)
            for slice_node in slice_nodes:
                if len(slice_node.users) == 0:
                    gm.graph.erase_node(slice_node)
    return changed


def fuse_slice_cat_with_mixed_input(gm: fx.GraphModule):
    changed = False
    for node in reversed(gm.graph.nodes):
        matched, src_nodes, axis, slice_param = _is_slice_cat_mixed_input(node)
        if matched:
            with gm.graph.inserting_before(node):
                slice_node = gm.graph.call_module(
                    "mlu_triton_fuse_slice_cat_mixed_input_module",
                    args=(src_nodes, slice_param),
                )
            node.replace_all_uses_with(slice_node)
            changed = True
            slice_nodes = node.args[0]
            gm.graph.erase_node(node)
            for slice_node in slice_nodes:
                if len(slice_node.users) == 0:
                    gm.graph.erase_node(slice_node)
    return changed


def fuse_mixed_ops_and_cat(gm: fx.GraphModule):
    changed = False
    for node in reversed(gm.graph.nodes):
        changed1 = False
        cat_input_list = []
        if check_cat_op(node):
            if (node.args[1] != -1) & (node.args[1] != 1):
                continue
            slice_input = []
            slice_param = []
            n_list = []
            for n in node.args[0]:
                is_slice = False
                if check_slice_op(n):
                    if len(n.args[0].meta["tensor_meta"].shape) == 2:
                        if n.args[1] == 1:
                            slice_input.append(n.args[0])
                            slice_param.append((n.args[2], n.args[3]))
                            n_list.append(n)
                            is_slice = True
                if is_slice == False:
                    if len(slice_input) > 0:
                        if slice_input.count(slice_input[0]) == len(slice_input):
                            with gm.graph.inserting_before(node):
                                slice_node = gm.graph.call_module(
                                    "mlu_triton_fuse_slice_cat_same_input_module",
                                    args=(slice_input[0], slice_param),
                                )
                                cat_input_list.append(slice_node)
                                changed1 = True
                        else:
                            cat_input_list += n_list
                    cat_input_list.append(n)
                    slice_input = []
                    slice_param = []
                    n_list = []
            if len(slice_input) > 0:
                if slice_input.count(slice_input[0]) == len(slice_input):
                    with gm.graph.inserting_before(node):
                        slice_node = gm.graph.call_module(
                            "mlu_triton_fuse_slice_cat_same_input_module",
                            args=(slice_input[0], slice_param),
                        )
                        cat_input_list.append(slice_node)
                        changed1 = True
                else:
                    cat_input_list += n_list
            if changed1 == True:
                with gm.graph.inserting_before(node):
                    cat_node = gm.graph.call_module(
                        "mlu_triton_fused_cat", args=(cat_input_list, -1)
                    )
                node.replace_all_uses_with(cat_node)
                slice_nodes = node.args[0]
                for slice_node in slice_nodes:
                    if len(slice_node.users) == 0:
                        gm.graph.erase_node(slice_node)
                gm.graph.erase_node(node)
                changed = True

    return changed


def fuse_slice_stack_same_input(gm: fx.GraphModule):
    changed = False
    for node in reversed(gm.graph.nodes):
        matched, src_node, axis, slice_param = _is_slice_stack(node)
        if matched:
            with gm.graph.inserting_before(node):
                slice_node = gm.graph.call_module(
                    "mlu_triton_fuse_slice_cat_same_input_module",
                    args=(src_node, slice_param),
                )
            with gm.graph.inserting_before(node):
                stack_node = gm.graph.call_module(
                    "expand_transpose_module",
                    args=(slice_node, len(slice_param)),
                )
            node.replace_all_uses_with(stack_node)
            changed = True
    return changed


def fuse_mixed_ops_and_stack(gm: fx.GraphModule):
    changed = False
    for node in reversed(gm.graph.nodes):
        changed1 = False
        cat_input_list = []
        if check_stack_op(node):
            slice_input = []
            slice_param = []
            n_list = []
            for n in node.args[0]:
                is_slice = False
                if check_slice_op(n):
                    if len(n.args[0].meta["tensor_meta"].shape) == 2:
                        if n.args[1] == 1:
                            slice_input.append(n.args[0])
                            slice_param.append((n.args[2], n.args[3]))
                            n_list.append(n)
                            is_slice = True
                if is_slice == False:
                    if len(slice_input) > 0:
                        if slice_input.count(slice_input[0]) == len(slice_input):
                            with gm.graph.inserting_before(node):
                                slice_node = gm.graph.call_module(
                                    "mlu_triton_fuse_slice_cat_same_input_module",
                                    args=(slice_input[0], slice_param),
                                )
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
                    slice_input = []
                    slice_param = []
                    n_list = []
            if len(slice_input) > 0:
                if slice_input.count(slice_input[0]) == len(slice_input):
                    with gm.graph.inserting_before(node):
                        slice_node = gm.graph.call_module(
                            "mlu_triton_fuse_slice_cat_same_input_module",
                            args=(slice_input[0], slice_param),
                        )
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
                new_node = gm.graph.call_module(
                    "mlu_triton_fused_slice_cat_to_all_slice",
                    args=(new_nodes, slice_offsets[idx]),
                )
                ori_node.replace_all_uses_with(new_node)
    return changed


class FusedSlice(Pattern):
    def process(self, gm: fx.GraphModule):
        changed = False
        gm.add_submodule(
            "mlu_triton_fuse_slice_cat_same_input_module",
            FuseSliceCatSameInputReplacement(),
        )
        # gm.add_submodule(
        #     "mlu_triton_fuse_slice_cat_mixed_input_module",
        #     FuseSliceCatMixedInputReplacement(),
        # )
        gm.add_submodule("expand_transpose_module", ExpandTransReplacement())
        gm.add_submodule("mlu_triton_fused_slice_cat_to_all_slice", SliceReplacement())
        gm.add_submodule("mlu_triton_fused_cat", CatReplacement())
        gm.add_submodule("mlu_triton_fused_stack", MergeCatReplacement())

        # slice & cat, all the slice ops are spliting from the same tensor.
        changed = changed | fuse_slice_cat_with_same_input(gm)

        # slice & cat, the slice ops are spliting from different tensors.
        # changed = changed | fuse_slice_cat_with_mixed_input(gm)

        # slice & cat, the inputs of cat are mixed with slice and other ops.
        changed = changed | fuse_mixed_ops_and_cat(gm)

        # slice & stack, all the slice ops are spliting from the same tensor.
        changed = changed | fuse_slice_stack_same_input(gm)

        # slice & stack, the inputs of stack are mixed with slice and other ops.
        changed = changed | fuse_mixed_ops_and_stack(gm)

        # merge multiple cat ops into one
        changed = changed | fuse_multiple_cat(gm)

        gm.graph.lint()
        gm.recompile()
        return changed
