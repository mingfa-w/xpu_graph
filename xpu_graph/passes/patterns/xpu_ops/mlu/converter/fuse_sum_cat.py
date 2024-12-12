import torch
from torch import nn, fx
import torch_mlu
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.check_op import (
    check_cat_op,
    check_sum_op,
    check_stack_op,
)
from ..triton_kernel.triton_fused_sum_cat import (
    mlu_triton_fuse_sum_cat_2d,
    mlu_triton_fuse_sum_cat_3d,
)


class ConcatenateSumOperation2(nn.Module):
    def forward(self, inputs, sum_dim, concat_dim):
        if sum_dim != [1]:
            raise NotImplementedError("sum_dim must be 1")
        if concat_dim != -1:
            raise NotImplementedError("concat_dim must be -1")

        batch_size, _, feature_dim = inputs[0].shape
        device = inputs[0].device
        sequence_lengths = [tensor.shape[1] for tensor in inputs]
        max_sequence_length = max(sequence_lengths)

        lengths_tensor = torch.tensor(
            sequence_lengths, device=device, dtype=torch.int32
        )

        return mlu_triton_fuse_sum_cat_3d(
            inputs,
            batch_size,
            lengths_tensor,
            feature_dim,
            len(inputs),
            max_sequence_length,
        )


class ConcatenateSumOperation1(nn.Module):
    def forward(
        self, inputs, sum_dim, concat_mode, keep_dims, concat_dim, is_concat=True
    ):
        batch_size = inputs[0].shape[0]
        device = inputs[0].device
        sequence_lengths = [tensor.shape[1] for tensor in inputs]
        max_sequence_length = max(sequence_lengths)
        length_tensor = torch.tensor(sequence_lengths, dtype=torch.int32, device=device)
        if keep_dims is True and concat_dim != 0:
            output = mlu_triton_fuse_sum_cat_3d(
                inputs,
                batch_size,
                length_tensor,
                1,
                len(inputs),
                max_sequence_length,
            )
        else:
            output = mlu_triton_fuse_sum_cat_2d(
                inputs,
                length_tensor,
                batch_size,
                max_sequence_length,
            )
            if is_concat and concat_dim == -1:
                output = output.reshape([-1])
        return output


def _validate_sum_concat(gm, node, source_nodes, sum_dims, keep_dims, concat_dim):
    if len(source_nodes) < 2:
        return False, None

    batch_sizes = [n.meta["tensor_meta"].shape[0] for n in source_nodes]
    if not all(size == batch_sizes[0] for size in batch_sizes):
        return False, None

    tensor_shapes = [n.meta["tensor_meta"].shape for n in source_nodes]
    dims = [len(shape) for shape in tensor_shapes]

    if not all(dim == dims[0] for dim in dims) or not 1 < dims[0] <= 3:
        return False, None

    is_2d = dims[0] == 2
    if not (all(axis == sum_dims[0] for axis in sum_dims) and sum_dims[0] == [1]):
        return False, None

    if not all(keep_dim is True for keep_dim in keep_dims):
        return False, None

    has_keep_dims = bool(keep_dims)
    is_concat = check_cat_op(node)

    sum_mode = 0
    concat_mode = 0
    if is_concat:
        if is_2d and has_keep_dims:
            pass
        elif is_2d and not has_keep_dims and concat_dim == -1:
            concat_mode = 1
        elif not is_2d and not has_keep_dims and concat_dim == -1:
            sum_mode = 1
        else:
            return False, None
    else:  # stack
        if is_2d and not has_keep_dims:
            concat_mode = 1
            concat_dim = 0
        else:
            return False, None

    if sum_mode == 0:
        with gm.graph.inserting_before(node):
            new_node = gm.graph.call_module(
                "mlu_triton_fused_cat_sum_1_replacement",
                args=(
                    source_nodes,
                    sum_dims[0],
                    concat_mode,
                    has_keep_dims,
                    concat_dim,
                    is_concat,
                ),
            )
        return True, new_node
    else:
        with gm.graph.inserting_before(node):
            new_node = gm.graph.call_module(
                "mlu_triton_fused_cat_sum_2_replacement",
                args=(
                    source_nodes,
                    sum_dims[0],
                    concat_dim,
                ),
            )
        return True, new_node
    return False, None


def process_match_sum_cat(gm: fx.GraphModule):
    changed = False
    for node in reversed(gm.graph.nodes):
        concat_dim = 1
        if check_cat_op(node) or check_stack_op(node):
            is_concat = check_cat_op(node)
            is_replace = False
            concat_input = []
            if is_concat:
                concat_dim = node.args[1]
                if node.meta == {}:
                    continue
                if len(node.meta["tensor_meta"].shape) - 1 == concat_dim:
                    concat_dim = -1
            if len(node.args[0]) > 1:
                n_list = []
                sum_dims = []
                source_nodes = []
                keep_dims = []
                for n in node.args[0]:
                    if check_sum_op(n):
                        n_list.append(n)
                        source_nodes.append(n.args[0])
                        sum_dims.append(n.args[1])
                        # keepdim
                        if len(n.args) == 3:
                            keep_dims.append(n.args[2])
                    else:
                        match_, new_node = _validate_sum_concat(
                            gm, node, source_nodes, sum_dims, keep_dims, concat_dim
                        )
                        if match_:
                            is_replace = True
                            concat_input.append(new_node)
                        else:
                            concat_input += n_list
                        concat_input.append(n)
                        n_list = []
                        sum_dims = []
                        source_nodes = []
                        keep_dims = []
                match_, new_node = _validate_sum_concat(
                    gm, node, source_nodes, sum_dims, keep_dims, concat_dim
                )
                if match_:
                    is_replace = True
                    concat_input.append(new_node)
                else:
                    concat_input += n_list

            if is_replace is True:
                changed = True
                if len(concat_input) == 1:
                    node.replace_all_uses_with(concat_input[0])
                else:
                    with gm.graph.inserting_before(node):
                        concat_node = gm.graph.create_node(
                            op="call_function",
                            target=torch.ops.aten.cat.default,
                            args=(concat_input, concat_dim),
                            name=node.name + "_1",
                        )
                    node.replace_all_uses_with(concat_node)
                gm.graph.erase_node(node)
    return changed


class FusedCatSum(Pattern):
    def process(self, gm: fx.GraphModule):
        changed = False
        gm.add_submodule(
            "mlu_triton_fused_cat_sum_1_replacement", ConcatenateSumOperation1()
        )
        gm.add_submodule(
            "mlu_triton_fused_cat_sum_2_replacement", ConcatenateSumOperation2()
        )

        changed = changed | process_match_sum_cat(gm)

        gm.graph.lint()
        gm.recompile()
        return changed
