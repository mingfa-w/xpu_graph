from torch import nn, fx
import torch
import torch_mlu
from typing import Optional
from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.utils import logger
from ...utils.match_sub_list import match_sub_list
from ...utils.check_ops import (
    check_cat_op,
    check_sum_op,
    check_meta_2d,
    check_mul_op,
    get_shape,
)
from .combo_cat_utils import find_longest_same_shape_sequence, find_longest_same_param
from xpu_graph.fx_utils import FxStage
import operator

MAX_INT64 = 9223372036854775807

MINI_LEN = 3


def match_mul(val):
    if not check_mul_op(val):
        return False
    # if "mul_replacement" in val.name
    #    # shape infer
    #    return False
    users = val.users
    if len(users) > 2:
        return False
    if len(users) == 2:
        a, b = list(users.keys())
        if not (a.op == "output" or b.op == "output"):
            return False
    inp1, inp2 = val.args
    if isinstance(inp1, (float, int)):
        return False
    if isinstance(inp2, (float, int)):
        return False
    if get_shape(val.args[0]) != get_shape(val.args[1]):
        return False
    return True


def match_mul1(val):
    if not check_mul_op(val):
        return False
    # if "mul_replacement" in val.name
    #    # shape infer
    #    return False
    users = val.users
    if len(users) > 2:
        return False
    if len(users) == 2:
        a, b = list(users.keys())
        if not (a.op == "output" or b.op == "output"):
            return False
    inp1, inp2 = val.args
    if isinstance(inp1, (float, int)):
        return False
    if not isinstance(inp2, (float, int)):
        return False
    return True


class ComboMulCat(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False

        candidates = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function"
            and (node.target == torch.ops.aten.cat.default)
        ]

        for node in candidates:
            ori_cat_input = node.args[0]
            axis = node.args[1]
            if len(ori_cat_input) < MINI_LEN:
                continue
            start, end = match_sub_list(
                ori_cat_input,
                match_mul,
            )
            if end - start + 1 < MINI_LEN:
                continue
            best_start, best_end = find_longest_same_shape_sequence(
                ori_cat_input, start, end, MINI_LEN
            )
            shape_length = best_end - best_start + 1
            if shape_length < MINI_LEN:
                continue
            n_list = ori_cat_input[best_start : best_end + 1]

            mul_inputs = []
            for mul_node in n_list:
                mul_inputs += [mul_node.args[0], mul_node.args[1]]

            new_cat_input = ori_cat_input[:best_start]
            with graph_module.graph.inserting_before(node):
                # replace_node = graph_module.graph.call_module(
                #    "mlu_triton_mul_cat_replacement",
                #    args=(mul_inputs, axis),
                # )
                # === 替换为call_function操作序列 ===

                # 步骤1: 重新排列mul_inputs - 偶数索引在前，奇数索引在后
                even_tensors = [mul_inputs[i] for i in range(0, len(mul_inputs), 2)]
                odd_tensors = [mul_inputs[i] for i in range(1, len(mul_inputs), 2)]
                reordered_inputs = even_tensors + odd_tensors

                # 步骤2: 1个cat操作 - 连接所有重新排列的张量
                fused_tensor = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.cat.default,
                    args=(reordered_inputs, axis),
                    name=node.name + "_mul_cat_replacement1",
                )

                # 步骤3: 计算分割点（需要知道单个张量的大小）
                # 假设所有张量在cat维度上大小相同，取第一个张量的大小
                num_pairs = len(mul_inputs) // 2

                # 获取第一个张量在指定轴上的大小
                # 这里需要根据实际情况获取shape信息
                # 方法1: 如果有shape信息
                single_tensor_size = get_shape(mul_inputs[0])[
                    axis
                ]  # 假设get_shape函数存在
                split_point = num_pairs * single_tensor_size

                # 步骤4: 第一个slice - 取前半部分（偶数索引对应的张量）
                left_tensor = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.slice.Tensor,
                    args=(fused_tensor, axis, 0, split_point),
                    name=node.name + "_left_slice",
                )

                # 步骤5: 第二个slice - 取后半部分（奇数索引对应的张量）
                right_tensor = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.slice.Tensor,
                    args=(fused_tensor, axis, split_point, MAX_INT64),
                    name=node.name + "_right_slice",
                )

                # 步骤6: 元素相乘
                replace_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.mul.Tensor,
                    args=(left_tensor, right_tensor),
                    name=node.name + "_mul_cat_replacement2",
                )

                # 步骤7: 处理n_list中有其他使用者的mul节点
                muls_with_other_users = []
                for i, mul_node in enumerate(n_list):
                    if len(mul_node.users) != 1:  # 不只被cat使用
                        muls_with_other_users.append((i, mul_node))

                if muls_with_other_users != []:
                    single_mul_size = get_shape(n_list[0])[axis]
                    with graph_module.graph.inserting_after(replace_node):
                        split_results = graph_module.graph.create_node(
                            op="call_function",
                            target=torch.ops.aten.split.Tensor,
                            args=(replace_node, single_mul_size, axis),
                            name=node.name + "_split_results",
                        )
                    with graph_module.graph.inserting_after(split_results):
                        for i, mul_node in muls_with_other_users:
                            split_part = graph_module.graph.call_function(
                                operator.getitem,
                                args=(split_results, i),
                                kwargs={},
                            )
                            mul_node.replace_all_uses_with(split_part)

            new_cat_input.append(replace_node)
            new_cat_input += ori_cat_input[best_end + 1 :]

            if len(new_cat_input) > 1:
                with graph_module.graph.inserting_before(node):
                    cat_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.cat.default,
                        args=(new_cat_input, axis),
                        name=node.name + "_replacement",
                    )
                node.replace_all_uses_with(cat_node)
            else:
                node.replace_all_uses_with(replace_node)
            changed = True

        return changed


class FusedMultConstCat(Pattern):
    _opt_level = OptLevel.level2

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False

        candidates = [
            node
            for node in graph_module.graph.nodes
            if node.op == "call_function"
            and (node.target == torch.ops.aten.cat.default)
        ]

        for node in candidates:
            ori_cat_input = node.args[0]
            axis = node.args[1]
            if len(ori_cat_input) < MINI_LEN:
                continue
            start, end = match_sub_list(
                ori_cat_input,
                match_mul1,
            )
            if end - start + 1 < MINI_LEN:
                continue
            best_start, best_end = find_longest_same_shape_sequence(
                ori_cat_input, start, end, MINI_LEN
            )
            shape_length = best_end - best_start + 1
            if shape_length < MINI_LEN:
                continue
            best_start, best_end = find_longest_same_param(
                ori_cat_input, best_start, best_end, MINI_LEN
            )
            shape_length = best_end - best_start + 1
            if shape_length < MINI_LEN:
                continue
            n_list = ori_cat_input[best_start : best_end + 1]

            mul_inputs = [n.args[0] for n in n_list]

            new_cat_input = ori_cat_input[:best_start]
            with graph_module.graph.inserting_before(node):
                fused_tensor = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.cat.default,
                    args=(mul_inputs, axis),
                    name=node.name + "_mulconst_cat_replacement1",
                )

                replace_node = graph_module.graph.create_node(
                    op="call_function",
                    target=torch.ops.aten.mul.Tensor,
                    args=(fused_tensor, n_list[0].args[1]),
                    name=node.name + "_mulconst_cat_replacement2",
                )
                # 步骤7: 处理n_list中有其他使用者的mul节点
                muls_with_other_users = []
                for i, mul_node in enumerate(n_list):
                    if len(mul_node.users) != 1:  # 不只被cat使用
                        muls_with_other_users.append((i, mul_node))

                if muls_with_other_users != []:
                    single_mul_size = get_shape(n_list[0])[axis]
                    with graph_module.graph.inserting_after(replace_node):
                        split_results = graph_module.graph.create_node(
                            op="call_function",
                            target=torch.ops.aten.split.Tensor,
                            args=(replace_node, single_mul_size, axis),
                            name=node.name + "_split_results",
                        )
                    with graph_module.graph.inserting_after(split_results):
                        for i, mul_node in muls_with_other_users:
                            split_part = graph_module.graph.call_function(
                                operator.getitem,
                                args=(split_results, i),
                                kwargs={},
                            )
                            mul_node.replace_all_uses_with(split_part)

            new_cat_input.append(replace_node)
            new_cat_input += ori_cat_input[best_end + 1 :]

            if len(new_cat_input) > 1:
                with graph_module.graph.inserting_before(node):
                    cat_node = graph_module.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.cat.default,
                        args=(new_cat_input, axis),
                        name=node.name + "_replacement",
                    )
                node.replace_all_uses_with(cat_node)
            else:
                node.replace_all_uses_with(replace_node)
            changed = True

        return changed
