from typing import Optional, Tuple, Union

import torch
from torch import nn, fx
import torch_mlu
from xpu_graph.utils import logger
from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from ...utils.submodule_manager import register_new_submodule
from ...utils.check_ops import (
    check_mm_op,
    check_add_op,
    check_view,
    check_act_op,
    check_trans_op,
    check_bmm_op,
    check_addmm_op,
    check_t_op,
    # get_shape,
)
import operator
from .triton_kernel.fused_groupgemm import fused_grouped_gemm

TensorShape = Union[torch.Size, Tuple[int, ...]]
NodeType = fx.Node
COMBINE_LEN = 4

from typing import Optional, Union
from torch.fx.node import map_arg


def extract_nodes_from_args_kwargs(args, kwargs):
    """
    从给定的 args 和 kwargs 中递归提取所有 fx.Node 实例。
    """
    nodes = []

    def recurse(item):
        if isinstance(item, fx.Node):
            nodes.append(item)
        elif isinstance(item, (list, tuple)):
            for elem in item:
                recurse(elem)
        elif isinstance(item, dict):
            for value in item.values():
                recurse(value)
        # 其他类型（如 int、float、str 等）不处理

    recurse(args)
    recurse(kwargs)
    return nodes


def get_ancestors(node):
    stack = [node]
    ancestors = []
    while stack:
        node = stack.pop()
        if node in ancestors:
            continue
        if node is None:
            continue
        if node.op == "placeholder":
            continue
        ancestors.append(node)
        stack += extract_nodes_from_args_kwargs(node.args, node.kwargs)
    if len(ancestors) > 0:
        # remove node
        ancestors = ancestors[1:]
    return ancestors


def get_shape(node: fx.Node):
    if not hasattr(node, "meta"):
        return None
    node_meta = node.meta
    if "val" not in node_meta:
        return None
    val = node_meta["val"]
    if not hasattr(val, "shape"):
        return None
    return val.shape


class MMNodeDesc:
    """
    NodeDesc class describes a node in torch.fx graph along with its associated information,
    including input1, input2, bias, and their shapes. It can also store an activation type.
    """

    def __init__(self) -> None:
        # The fx.Node itself (typically an mm or addmm operation)
        self.node: Optional[NodeType] = None
        self.input1: Optional[NodeType] = None
        self.input2: Optional[NodeType] = None
        self.bias: Optional[Union[NodeType, int, float]] = None
        self.input1_shape: Optional[TensorShape] = None
        self.input2_shape: Optional[TensorShape] = None
        self.bias_shape: Optional[TensorShape] = None
        # Activation function string (default "none")
        self.act: str = "none"
        self.input1_ancestors = []
        self.input2_ancestors = []
        self.bias_ancestors = []

    def set_node(self, node):
        self.node = node

    def set_input1(self, input1):
        self.input1 = input1
        self.input1_shape = get_shape(input1)
        self.input1_ancestors = get_ancestors(self.input1)

    def set_input2(self, input2):
        self.input2 = input2
        self.input2_shape = get_shape(input2)
        self.input2_ancestors = get_ancestors(self.input2)

    def set_bias(self, bias):
        self.bias = bias
        if bias is not None:
            self.bias_shape = get_shape(bias)
            self.bias_ancestors = get_ancestors(self.bias)

    def set_act(self, act: str):
        self.act = act


def get_node_desc(node):
    intpu1 = None
    intpu2 = None
    bias = None
    act = None
    check_args = False
    if node.target == torch.ops.aten.mm.default:
        input1 = node.args[0]
        input2 = node.args[1]
    elif node.target == torch.ops.aten.addmm.default:
        bias = node.args[0]
        input1 = node.args[1]
        input2 = node.args[2]
    else:
        # fused_tmo_xxx
        input1 = node.args[0]
        input2 = node.args[2]
        bias = node.args[5]
        # TODO(JYJ):Remove restrictions
        trans_b = node.args[4]
        if trans_b == True:
            return None
        if isinstance(bias, (int, float)):
            return None
        if "bmm" in node.target:
            act = node.args[9]
        else:
            act = node.args[6]

    if get_shape(input1) == None:
        return None
    if get_shape(input2) == None:
        return None
    if bias is not None:
        if get_shape(bias) == None:
            return None

    mm_desc = MMNodeDesc()
    mm_desc.set_node(node)
    mm_desc.set_input1(input1)
    mm_desc.set_input2(input2)
    mm_desc.set_bias(bias)
    mm_desc.set_act(act)
    # print("mm_desc:", mm_desc.node)
    # print(mm_desc.input1_ancestors)
    # print(mm_desc.input2_ancestors)
    # print(mm_desc.bias_ancestors)
    return mm_desc


def all_same_tensor(tensor_list):
    if not tensor_list:
        return True
    first = tensor_list[0]
    return all(t is first for t in tensor_list)


class FusedCombineBmm(nn.Module):
    def __init__(self, batch_sizes):
        super().__init__()
        device = torch.mlu.current_device()
        self.batch_tensor = torch.tensor(
            batch_sizes, dtype=torch.int64, device="cpu"
        )

    def forward(self, input_list, weight_list, bias_list, batch_sizes, act: str):
        output = None
        #print("#############len(input_list[0].shape): ", len(input_list[0].shape))
        if len(input_list[0].shape) == 3:
            output = self.forward_bmm(input_list, weight_list, bias_list, act)
        else:
            output = self.forward_mm(input_list, weight_list, bias_list, batch_sizes, act)

        bias = bias_list[0]
        if bias is None:
            bias_batch = None
        else:
            bias_list = [
                bias.unsqueeze(0) if len(bias.shape) == 1 else bias
                for bias in bias_list
            ]
            bias_batch = torch.stack(
                bias_list, dim=0
            )  # Stack all biases along a new dimension
        # Add bias batch if available
        if bias_batch is not None:
            output = output + bias_batch

        # Apply activation function if specified
        if act == "relu":
            output = torch.relu(output)
        elif act == "gelu":
            output = torch.gelu(output)
        elif act == "silu":
            output = torch.silu(output)
        elif act == "sigmoid":
            output = torch.sigmoid(output)

        return output

    def forward_bmm(self, input_list, weight_list, bias_list, act: str):

        input_batch = torch.stack(input_list, dim=0)
        weight_batch = torch.stack(weight_list, dim=0)
        T, B, K, N = weight_batch.shape
        M = input_list[0].shape[1]
        output = torch.bmm(
            input_batch.view(-1, M, K), weight_batch.view(-1, K, N)
        ).view(T, B, M, N)
        return output

    def forward_mm(self, input_list, weight_list, bias_list, batch_sizes, act: str):
        #if all_same_tensor(input_list):
        #    input = input_list[0]
        #    input_batch = input.unsqueeze(0).expand(
        #        len(input_list), input.shape[0], input.shape[1]
        #    )
        #else:
        #    input_batch = torch.stack(input_list, dim=0)
        input_batch = torch.cat(input_list, dim=0)
        weight_batch = torch.stack(weight_list, dim=0)
        #B, K, N = weight_batch.shape

        batch_sizes_tensor = torch.tensor(
            batch_sizes, dtype=torch.int64, device="cpu",
        )
        #output_batch = fused_grouped_gemm(input_batch, weight_batch, batch_sizes_tensor, trans_b=False)
        output_batch = fused_grouped_gemm(input_batch, weight_batch, self.batch_tensor, trans_b=False)
        output = output_batch.view(len(batch_sizes), batch_sizes[0], -1)
        #output = torch.stack(output_list, dim=0)
        #for i in range(len(tmp)):
        #    print("######tmp[i].shape: ", tmp[i].shape)
        #print("")
        #print("")

        #input_batch = torch.stack(input_list, dim=0)
        #output = torch.bmm(
        #    input_batch,
        #    weight_batch,  # [B, M, N]
        #)
        return output

        """
        # Fallback case: ordinary matmul + bias for each pair
        else:
            if len(input_list) == 1:
                input_list = input_list * len(weight_list)
            output = [
                torch.matmul(input, weight) + bias
                for input, weight, bias in zip(input_list, weight_list, bias_list)
            ]
        """


def find_last_node_in_list(gm: fx.GraphModule, node_list: list[fx.Node]) -> fx.Node:
    """
    Given a list of nodes, find the one that appears last in the graph's topological order.
    """
    node_set = set(node_list)  # Faster lookup
    last_node = None

    for node in gm.graph.nodes:
        if node in node_set:
            last_node = node  # Update when we see a matching node

    return last_node


def partly_topo_sort(gm: fx.Graph, node: fx.Node):
    import queue

    que = queue.Queue()
    que.put(node)
    while not que.empty():
        cur = que.get()
        for user in cur.users:
            if user < cur:
                cur.append(user)
                que.put(user)


def replace_node(graph_module, nodes):
    new_input = [n.input1 for n in nodes]
    new_weight = [n.input2 for n in nodes]
    new_bias = [n.bias for n in nodes]
    act = nodes[0].act
    batch_sizes = [new_input[0].meta["val"].shape[0]] * len(new_input)

    if len(new_weight) < COMBINE_LEN:
        return
    with graph_module.graph.inserting_after(
        find_last_node_in_list(graph_module, new_input + new_weight + new_bias)
    ):
        module_name = register_new_submodule(
            graph_module,
            "fused_combine_bmm",
            FusedCombineBmm,
            args=(batch_sizes,),
        )
        new_node = graph_module.graph.call_module(
            module_name,
            #"fused_combine_bmm",
            args=(new_input, new_weight, new_bias, batch_sizes, act),
        )
    with graph_module.graph.inserting_after(new_node):
        for idx, n in enumerate(nodes):
            new_n = graph_module.graph.call_function(
                operator.getitem, args=(new_node, idx)
            )
            n.node.replace_all_uses_with(new_n)
            partly_topo_sort(graph_module, new_n)
    graph_module.graph.lint()
    graph_module.recompile()


def has_dependency(a, b):
    all_a = a.input1_ancestors + a.input2_ancestors + a.bias_ancestors
    all_b = b.input1_ancestors + b.input2_ancestors + b.bias_ancestors
    return (
        a.input1 in all_b
        or a.input2 in all_b
        or (a.bias in all_b if a.bias else False)
        or b.input1 in all_a
        or b.input2 in all_a
        or (b.bias in all_a if b.bias else False)
    )


def find_dep(graph_module, nodes):
    groups = []

    for node in nodes:
        placed = False
        for group in groups:
            if any(has_dependency(node, other) for other in group):
                continue
            group.append(node)
            placed = True
            break
        if not placed:
            groups.append([node])

    return groups


def combine_matmul(graph_module, candidates):
    changed = False
    group_by_shape = {}
    for n in candidates:
        mm_desc = get_node_desc(n)
        if mm_desc == None:
            continue
        key = (
            mm_desc.input1_shape,
            mm_desc.input2_shape,
            mm_desc.bias_shape,
            mm_desc.act,
        )
        if key not in group_by_shape:
            group_by_shape[key] = []
        group_by_shape[key].append(mm_desc)

    for key1, group_nodes in group_by_shape.items():
        group_by_input = find_dep(graph_module, group_nodes)
        for nodes in group_by_input:
            if len(nodes) < COMBINE_LEN:
                continue
            changed = True
            replace_node(graph_module, nodes)
    return changed


class FusedCombineMatMul(Pattern):
    _opt_level = OptLevel.level2
    _pattern_group = PatternGroup.GROUP1

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False
        print(graph_module.graph)

        #graph_module.add_submodule("fused_combine_bmm", FusedCombineBmm())
        target_module = [
            "mlu_tmo_fused_bmm_replacement",
            torch.ops.aten.mm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.addmm.default,
            "mlu_tmo_fused_matmul_replacement",
            "mlu_tmo_fused_matmul_add_replacement",
            "mlu_tmo_fused_matmul_act_replacement",
            "mlu_tmo_fused_matmul_add_act_replacement",
            "mlu_tmo_fused_bmm_add_replacement",
            "mlu_tmo_fused_bmm_act_replacement",
            "mlu_tmo_fused_bmm_add_act_replacement",
        ]

        for module in target_module:
            candidates = [
                node
                for node in graph_module.graph.nodes
                if (node.op == "call_function" or node.op == "call_module")
                and node.target == module
            ]
            if len(candidates) < COMBINE_LEN:
                continue
            changed = changed | combine_matmul(graph_module, candidates)
            #graph_module.graph.lint()
            #graph_module.recompile()
        #if changed:
        #    print("after")
        #    print(graph_module.graph)

        return changed
