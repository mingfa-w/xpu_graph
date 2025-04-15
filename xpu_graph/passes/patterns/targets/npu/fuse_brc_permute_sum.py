import torch
from torch import nn, fx
from torch.fx.node import Node
import torch.fx as fx
from typing import Optional, Tuple, List
import torch_npu
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.config import OptLevel
from ...utils.check_ops import (
    get_dtype,
    check_slice_op,
    check_unsqueeze_op,
    check_expand_op,
    check_npu_dtype_cast_op,
    check_rsub_scalar_op,
    check_mask_fill_op,
    check_add_op,
    check_mul_op,
    check_trans_op,
    check_softmax_op,
)

from .triton_kernel.fused_brc_permute_sum import fused_brc_permute_sum

class BrcPermuteSumOperation(nn.Module):
    def forward(self, view_7, buf47, buf59, arg107_1, buf61):
        return torch.ops.torch_npu_triton.fused_brc_permute_sum(view_7, buf47, buf59, arg107_1, buf61)

class FusedBrcPermuteSum(Pattern):
    _opt_level = OptLevel.level2

    def _match_expand_div_pair(self, div_node: Node) -> Optional[Tuple[Node, Node]]:
        if div_node.op != "call_function" or div_node.target != torch.ops.aten.div.Tensor:
            return None

        _, divisor = div_node.args
        if divisor.op != "call_function" or divisor.target != torch.ops.aten.expand.default:
            return None

        base_input = divisor.args[0]
        expand_shape = divisor.args[1]
        return (divisor, base_input, expand_shape)

    # TODO: currently we do not parse the const value inside the pattern,
    # i.e. 1.0 and 0.07216878364870322 shown in the fused triton kernel.
    def _match_pattern(self, root: Node) -> Optional[List[Node]]:
        # slice
        if not check_slice_op(root):
            return None
        if not (get_dtype(root.args[0]) in (torch.int64, torch.int32)):
            return None
        slice = root
        #
        input0 = None
        input1 = None
        input2 = None
        input3 = slice.args[0]
        input4 = None
        # slice -> unqueeze
        slice_unqueeze = None
        for user in slice.users:
            if not check_unsqueeze_op(user):
                continue
            if not (user.args[1] == 1):
                continue
            slice_unqueeze = user
            break
        if (slice_unqueeze == None):
            return None
        # slice -> unqueeze -> unqueeze
        slice_unqueeze2 = None
        for user in slice_unqueeze.users:
            if not check_unsqueeze_op(user):
                continue
            if not (user.args[1] == 2):
                continue
            slice_unqueeze2 = user
            break
        if (slice_unqueeze2 == None):
            return None
        # slice -> unqueeze -> unqueeze -> expand
        slice_unqueeze2_expand = None
        for user in slice_unqueeze2.users:
            if not check_expand_op(user):
                continue
            if not ((len(user.args[1]) == 4) and (user.args[1][1] == 1)):
                continue
            slice_unqueeze2_expand = user
            break
        if (slice_unqueeze2_expand == None):
            return None
        # slice -> unqueeze -> unqueeze -> expand -> cast
        slice_unqueeze2_expand_cast = None
        for user in slice_unqueeze2_expand.users:
            if not check_npu_dtype_cast_op(user):
                continue
            if not (user.args[1] == torch.float16):
                continue
            slice_unqueeze2_expand_cast = user
            break
        if (slice_unqueeze2_expand_cast == None):
            return None
        # slice -> unqueeze -> unqueeze -> expand -> cast -> rsub
        slice_unqueeze2_expand_cast_rsub = None
        for user in slice_unqueeze2_expand_cast.users:
            if not check_rsub_scalar_op(user):
                continue
            if not (user.args[1] == 1.0):
                continue
            slice_unqueeze2_expand_cast_rsub = user
            break
        if (slice_unqueeze2_expand_cast_rsub == None):
            return None
        # slice -> unqueeze -> unqueeze -> expand -> cast -> rsub -> cast
        slice_unqueeze2_expand_cast_rsub_cast = None
        for user in slice_unqueeze2_expand_cast_rsub.users:
            if not check_npu_dtype_cast_op(user):
                continue
            if not (user.args[1] == torch.bool):
                continue
            slice_unqueeze2_expand_cast_rsub_cast = user
            break
        if (slice_unqueeze2_expand_cast_rsub_cast == None):
            return None
        # slice -> unqueeze -> unqueeze -> expand -> cast -> rsub -> cast -> masked_fill
        slice_unqueeze2_expand_cast_rsub_mskfil = None
        for user in slice_unqueeze2_expand_cast_rsub.users:
            if not check_mask_fill_op(user):
                continue
            if not ((user.args[0] == slice_unqueeze2_expand_cast_rsub) and (user.args[1] == slice_unqueeze2_expand_cast_rsub_cast)):
                continue
            slice_unqueeze2_expand_cast_rsub_mskfil = user
            break
        if (slice_unqueeze2_expand_cast_rsub_mskfil == None):
            return None
        input4 = slice_unqueeze2_expand_cast_rsub_mskfil.args[2]
        # slice -> unqueeze -> unqueeze -> expand -> cast -> rsub -> cast -> masked_fill
        # -> add
        add = None
        add_other = None
        for user in slice_unqueeze2_expand_cast_rsub_mskfil.users:
            if not check_add_op(user):
                continue
            add = user
            add_other = next((e for e in user.args if e != slice_unqueeze2_expand_cast_rsub_mskfil), None)
            break
        if (add == None):
            return None
        if (add_other == None):
            return None
        # -> add <- add
        if not check_add_op(add_other):
            return None
        add_add = add_other
        # -> add <- add <- mul
        add_add_mul = None
        for arg in add_add.args:
            if not check_mul_op(arg):
                input0 = arg
                continue
            # FIXME: Do we need to check the 2nd argument which is a scalar?
            add_add_mul = arg
            break
        # -> add <- add <- mul <- add
        add_add_mul_add = None
        for arg in add_add_mul.args:
            if not check_add_op(arg):
                continue
            add_add_mul_add = arg
            break
        if (add_add_mul_add == None):
            return None
        # -> add <- add <- mul <- add <- transpose
        add_add_mul_add_trans = None
        for arg in add_add_mul_add.args:
            if not check_trans_op(arg):
                input1 = arg
                continue
            if not ((arg.args[1] == -1) and (arg.args[2] == -2)):
                continue
            add_add_mul_add_trans = arg
            break
        input2 = add_add_mul_add_trans.args[0]
        # -> add -> softmax
        add_softmax = None
        for user in add.users:
            if not check_softmax_op(user):
                continue
            if not (user.args[1] == -1):
                continue
            add_softmax = user
            break
        if (add_softmax == None):
            return None
        final = add_softmax
        return [
            input0, input1, input2, input3, input4, final
        ]

    def process(self, gm: fx.GraphModule):
        graph = gm.graph
        changed = False
        gm.add_submodule("npu_triton_fused_brc_permute_sum", BrcPermuteSumOperation())

        for node in list(graph.nodes):
            matched_nodes = self._match_pattern(node)
            if not matched_nodes:
                continue

            (input0, input1, input2, input3, input4, final) = matched_nodes
            with graph.inserting_before(final):
                fused_node = graph.call_module(
                    "npu_triton_fused_brc_permute_sum",
                    args=(input0, input1, input2, input3, input4),
                )
            final.replace_all_uses_with(fused_node)
            changed = True

            nodes_to_remove = [
                input0, input1, input2, input3, input4, final
            ]
            for node in nodes_to_remove:
                if isinstance(node, ()) and len(node.users) == 0:
                    graph.erase_node(node)

        if changed:
            gm.graph.lint()
            gm.recompile()
        return changed
