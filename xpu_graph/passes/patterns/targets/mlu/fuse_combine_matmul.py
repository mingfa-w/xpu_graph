from typing import Optional, Tuple, Union

import torch
from torch import nn, fx
import torch_mlu
from xpu_graph.utils import logger
from xpu_graph.config import OptLevel
from xpu_graph.passes.patterns.pattern import Pattern, PatternGroup
from ...utils.check_ops import (
    check_mm_op,
    check_add_op,
    check_view,
    check_act_op,
    check_trans_op,
    check_bmm_op,
    check_addmm_op,
    check_t_op,
    get_shape,
)
import operator

TensorShape = Union[torch.Size, Tuple[int, ...]]
NodeType = fx.Node
COMBINE_LEN = 4

from typing import Optional, Union


class MMNodeDesc:
    """
    NodeDesc class describes a node in torch.fx graph along with its associated information,
    including input, weight, bias, and their shapes. It can also store an activation type.
    """

    def __init__(self) -> None:
        # The fx.Node itself (typically an mm or addmm operation)
        self.node: Optional[NodeType] = None
        self.input: Optional[NodeType] = None
        self.weight: Optional[NodeType] = None
        self.weight_shape: Optional[TensorShape] = None
        self.bias: Optional[Union[NodeType, int, float]] = None
        self.bias_shape: Optional[TensorShape] = None
        # Activation function string (default "none")
        self.act: str = "none"

    def set_node(self, node):
        self.node = node

    def set_input(self, input):
        self.input = input

    def set_weight(self, weight):
        self.weight = weight
        if self.is_weight_args():
            self.weight_shape = get_shape(weight)

    def set_bias(self, bias):
        self.bias = bias
        if bias is not None and not self.is_bias_args():
            if isinstance(self.bias, (int, float)):
                self.bias_shape = ()
            else:
                self.bias_shape = get_shape(bias)

    def set_act(self, act: str):
        self.act = act

    def is_weight_args(self) -> bool:
        """
        Check if the weight node is a placeholder.
        """
        if self.weight is not None:
            return self.weight.op == "placeholder"
        else:
            return False

    def is_bias_args(self) -> bool:
        """
        Check if the bis node is a placeholder.
        If bias is an int or float, treat it as a valid bias (return True).
        """
        if self.bias is None:
            return True
        if isinstance(self.bias, (int, float)):
            return True
        if hasattr(self.bias, "op"):
            return self.bias.op == "placeholder"
        return True  # fallback safe


def get_node_desc(node):
    weight = None
    bias = None
    act = None
    check_args = False
    if node.target == torch.ops.aten.mm.default:
        weight = node.args[1]
    elif node.target == torch.ops.aten.addmm.default:
        bias = node.args[0]
        weight = node.args[2]
    else:
        # fused_tmo_xxx
        weight = node.args[2]
        bias = node.args[5]
        act = node.args[6]
        # TODO(JYJ):Remove restrictions
        trans_b = node.args[4]
        if trans_b == True:
            return None
        if isinstance(bias, (int, float)):
            return None

    mm_desc = MMNodeDesc()
    mm_desc.set_node(node)
    mm_desc.set_input(node.args[0])
    mm_desc.set_weight(weight)
    mm_desc.set_bias(bias)
    mm_desc.set_act(act)
    return mm_desc


class FusedCombineBmm(nn.Module):
    def forward(self, input_list, weight_list, bias_list, act: str):
        """
        Fuses multiple batched matrix multiplications with optional bias addition and activation.

        Args:
            input_list (Tensor): List of input tensor.
            weight_list (List[Tensor]): List of weight matrices.
            bias_list (List[Tensor or scalar]): List of biases corresponding to each weight.
            act (str): Activation function to apply ('relu', 'gelu', 'silu', or 'none').

        Returns:
            Tensor: Output tensor after batched mm/addmm and optional activation.
        """
        weight_batch = torch.stack(weight_list, dim=0) 
        #import pdb;pdb.set_trace()
        #print(weight_batch)
        B, K, N = weight_batch.shape

        bias = bias_list[0]
        if bias is None:
            bias_batch = None
        else:
            bias_list = [bias.unsqueeze(0) if len(bias.shape) == 1 else bias for bias in bias_list]
            bias_batch = torch.stack(
                bias_list, dim=0
            )  # Stack all biases along a new dimension

        if len(input_list) == 1:
            input = input_list[0]
            input_batch = input.unsqueeze(0).expand(
                    B, input.shape[0], input.shape[1]
                )
        else:
            input_batch = torch.stack(
                input_list, dim=0
            )

        #import pdb;pdb.set_trace()
        output = torch.bmm(
            input_batch, weight_batch,  # [B, M, N]
        )


        '''
        # Fallback case: ordinary matmul + bias for each pair
        else:
            if len(input_list) == 1:
                input_list = input_list * len(weight_list)
            output = [
                torch.matmul(input, weight) + bias
                for input, weight, bias in zip(input_list, weight_list, bias_list)
            ]
        '''

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

        '''
        # Case 2: input shape [T, B, M] and weight shape [T, B, K]
        elif len(weight_list[0].shape) == 3 and len(input.shape) == 3:
            weight_batch = torch.stack(weight_list, dim=0)  # [T, B, K, N]
            T, B, K, N = weight_batch.shape
            _, M, _ = input.shape

            input_ = (
                input.unsqueeze(0).expand(T, B, M, K).reshape(-1, M, K)
            )  # [T, B, M] -> [T*B, M, K]
            output = torch.bmm(input_, weight_batch.view(-1, K, N)).view(  # [T*B, K, N]
                T, B, M, N
            )  # [T, B, M, N]
        '''
        return output


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

def replace_node(graph_module, nodes, same_input):
    if same_input:
        new_input = [nodes[0].input]
    else:
        new_input = [n.input for n in nodes]
    new_weight = [n.weight for n in nodes]
    new_bias = [n.bias for n in nodes]
    act = nodes[0].act

    if len(new_weight) < COMBINE_LEN:
        return
    with graph_module.graph.inserting_after(find_last_node_in_list(graph_module, new_input)):
        new_node = graph_module.graph.call_module(
            "fused_combine_bmm",
            args=(new_input, new_weight, new_bias, act),
        )
    with graph_module.graph.inserting_after(new_node):
        for idx, n in enumerate(nodes):
            new_n = graph_module.graph.call_function(
                operator.getitem, args=(new_node, idx)
            )
            n.node.replace_all_uses_with(new_n)

def split_inp1(graph_module, nodes):
    if len(nodes) <= COMBINE_LEN:
        return []
    input_dict = {}
    for n in nodes:
        input_ = n.input
        if input_ not in input_dict:
            input_dict[input_] = [n]
        else:
            input_dict[input_].append(n)

    replaced_nodes = []
    for key, nodes in input_dict.items():
        if len(nodes) < COMBINE_LEN:
            continue
        replace_node(graph_module, nodes, True)
        replaced_nodes += nodes
    return replaced_nodes

def get_real_input(node):
    while True:
        if node.op == "placeholder":
            break
        elif node.target in [operator.getitem, torch.ops.aten.view.default, torch.ops.aten.squeeze.dim]:
            node = node.args[0]
        else:
            break
    return node

def split_inp_getitem(graph_module, nodes):
    if len(nodes) <= COMBINE_LEN:
        return []
    input_dict = {}
    for n in nodes:
        input_ = n.input
        src_node = get_real_input(input_) 
        input_shape = get_shape(input_)
        key = (src_node, input_shape)
        if key not in input_dict:
            input_dict[key] = [n]
        else:
            input_dict[key].append(n)

    replaced_nodes = []
    for key, nodes in input_dict.items():
        if len(nodes) < COMBINE_LEN:
            continue
        replace_node(graph_module, nodes, False)
        replaced_nodes += nodes
    return replaced_nodes


def combine_matmul(graph_module, candidates):
    changed = False
    # 1. split by weight_shape and bias_shape and act
    weight_shape_dict = {}
    for n in candidates:
        mm_desc = get_node_desc(n)
        if mm_desc == None:
            continue
        if not (mm_desc.is_bias_args() and mm_desc.is_weight_args()):
            continue
        key = (mm_desc.weight_shape, mm_desc.bias_shape, mm_desc.act)
        if key not in weight_shape_dict:
            weight_shape_dict[key] = []
        weight_shape_dict[key].append(mm_desc)

    # 2. split by input
    for key, nodes in weight_shape_dict.items():
        # a. the same input
        replaced_nodes = split_inp1(graph_module, nodes)
        if len(replaced_nodes) > 0:
            changed = True
            nodes = [x for x in nodes if x not in replaced_nodes]
        # b. split/slice
        replaced_nodes = split_inp_getitem(graph_module, nodes)
        if len(replaced_nodes) > 0:
            changed = True
            nodes = [x for x in nodes if x not in replaced_nodes]
        # c. TBD
    return changed


class FusedCombineMatMul(Pattern):
    _opt_level = OptLevel.level2
    _pattern_group = PatternGroup.GROUP1

    def process(self, graph_module: fx.GraphModule) -> bool:
        changed = False

        graph_module.add_submodule("fused_combine_bmm", FusedCombineBmm())
        #print("before")
        #print(graph_module.graph)
        target_module = [
            torch.ops.aten.mm.default,
            torch.ops.aten.addmm.default,
            "mlu_tmo_fused_matmul_1_replacement",
            "mlu_tmo_fused_matmul_2_replacement",
            "mlu_tmo_fused_matmul_3_replacement",
            "mlu_tmo_fused_matmul_4_replacement",
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
            graph_module.graph.lint()
            graph_module.recompile()
        if changed:
            print("after")
            print(graph_module.graph)

        return changed
