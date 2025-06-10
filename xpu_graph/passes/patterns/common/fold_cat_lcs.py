import torch
import torch.fx as fx
from xpu_graph.fx_utils import FxStage
from xpu_graph.passes.patterns.pattern import Pattern
from xpu_graph.passes.patterns.utils.check_ops import check_cat_op

from ..utils.check_ops import get_shape

MINI_LEN=3

def longest_common_subsequence(seq1, seq2):
    """
    计算两个序列的最长公共子序列

    Args:
        seq1: 第一个序列（列表）
        seq2: 第二个序列（列表）

    Returns:
        tuple: (最长公共子序列长度, 最长公共子序列)
    """
    m, n = len(seq1), len(seq2)

    # 创建DP表，dp[i][j]表示seq1前i个元素和seq2前j个元素的LCS长度
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 填充DP表
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                # 如果当前元素相同，LCS长度+1
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # 如果不同，取较大的LCS长度
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    # 回溯构造实际的LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[i - 1] == seq2[j - 1]:
            lcs.append(seq1[i - 1])
            i -= 1
            j -= 1
        elif dp[i - 1][j] > dp[i][j - 1]:
            i -= 1
        else:
            j -= 1

    # 因为是从后往前构造的，需要反转
    lcs.reverse()

    return dp[m][n], lcs


def find_sublist(main_list, sub_list):
    """
    在主列表中查找子列表的第一个起始位置

    Args:
        main_list: 主列表
        sub_list: 要查找的子列表

    Returns:
        int: 找到则返回起始索引，未找到返回-1
    """
    if not sub_list:
        return 0

    if len(sub_list) > len(main_list):
        return -1

    for i in range(len(main_list) - len(sub_list) + 1):
        if main_list[i : i + len(sub_list)] == sub_list:
            return i

    return -1


def find_axis(nodea):
    if len(nodea.args) == 1:
        axisa = 0
    else:
        axisa = nodea.args[1]
    if axisa == -1:
        axisa = len(get_shape(nodea)) - 1
    return axisa


class FoldCatLCS(Pattern):
    """
    cat1([op1, op2, op3, op4])
    cat2([op1, op2, op3, op5])
    ->
    cat0([op1, op2, op3])
    cat1([cat0, op4])
    cat2([cat0, op5])
    """

    def process(self, gm: fx.GraphModule):
        changed = False
        candidates = [
            node
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.target == torch.ops.aten.cat.default
            and len(node.args[0]) > 2
        ]
        lcs_list = []
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                nodea = candidates[i]
                nodeb = candidates[j]
                # axis
                if find_axis(nodea) != find_axis(nodeb):
                    continue
                length, lcs = longest_common_subsequence(nodea.args[0], nodeb.args[0])
                if length > MINI_LEN:
                    lcs_list.append((lcs, find_axis(nodea)))

        lcs_list.sort(key=lambda x: len(x[0]), reverse=True)  # 按LCS长度降序排序
        for lcs, axis in lcs_list:
            ca = []
            for node in candidates:
                idx = find_sublist(node.args[0], lcs)
                if idx != -1:
                    ca.append((node, idx))
            if len(ca) > 1:
                first_node = ca[0][0]
                with gm.graph.inserting_before(first_node):
                    new_cat = gm.graph.create_node(
                        op="call_function",
                        target=torch.ops.aten.cat.default,
                        args=(lcs, axis),
                        name=first_node.name + "_subcat",
                    )
                for node, idx in ca:
                    new_cat_input = (
                        node.args[0][0:idx] + [new_cat] + node.args[0][idx + len(lcs) :]
                    )

                    with gm.graph.inserting_before(node):
                        new_node = gm.graph.create_node(
                            op="call_function",
                            target=torch.ops.aten.cat.default,
                            args=(new_cat_input, axis),
                            name=node.name + "_optimized",
                        )

                    node.replace_all_uses_with(new_node)
                    changed = True

        return changed
