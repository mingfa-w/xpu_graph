from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import fx

from xpu_graph.utils import logger

from .check_ops import check_t_op, check_trans_op


class DefaultSliceSumCatModule(torch.nn.Module):
    def __init__(self, slice_param):
        """
        Args:
            slice_param (list of tuples): A list of slice indices, where each tuple
                                          contains (start_idx, end_idx) for slicing.
        """
        super().__init__()

        slice_ = []
        for param in slice_param:
            slice_ += [param[0], param[1]]

        self.slice_param_list = slice_param

    def forward(self, input):
        """
        Forward pass for the SliceSumCatOperation.

        Args:
            input (torch.Tensor): The input tensor of shape (batch, row, col).

        Returns:
            torch.Tensor: The output tensor of shape (batch, len(slice_param) * col). The processed tensor after slice -> sum -> cat operations.
        """
        target_tensors = []
        for slice_arg in self.slice_param_list:
            slice_tensor = input[:, slice_arg[0] : slice_arg[1], :]
            sum_tensor = torch.sum(slice_tensor, dim=[1])
            target_tensors.append(sum_tensor)
        return torch.cat(target_tensors, axis=-1)


class DenseParams:
    def __init__(self) -> None:
        self.input: Optional[fx.Node] = None
        self.weight: Optional[fx.Node] = None
        self.weight_trans: bool = False
        self.bias: Optional[Union[fx.Node, int, float]] = None
        self.act: str = "none"
        self.node_name: list = []

    def set_params(self, input, weight, weight_trans, bias, act):
        self.input = input
        self.weight = weight
        self.weight_trans = weight_trans
        self.bias = bias
        self.act = act
        self.check_shape()
        return self

    def set_node(self, node):
        if len(node.args) != 5:
            return False
        self.input = node.args[0]
        self.weight = node.args[1]
        self.weight_trans = node.args[2]

        if node.args[3] is not None:
            if not self.set_bias(node.args[3]):
                return False

        if not self.set_act(node.args[4]):
            return False

        return True

    def set_weight(self, node):
        if check_trans_op(node):
            trans_param = (node.args[1], node.args[2])
            if trans_param in [(0, 1), (1, 0)]:
                self.weight_trans = True
                node = node.args[0]
            else:
                return False
        elif check_t_op(node):
            self.weight_trans = True
            node = node.args[0]
        weight_shape = node.meta["val"].shape
        if len(weight_shape) != 2:
            logger.warning(f"MatMul pass: Unsupported weight dim {weight_shape}")
            return False
        self.weight = node

        if self.input:
            return self.check_shape()
        return True

    def set_input(self, node):
        input_shape = node.meta["val"].shape
        if len(input_shape) != 2:
            logger.warning(f"MatMul pass: Unsupported input dim {input_shape}")
            return False
        self.input = node

        if self.weight:
            return self.check_shape()
        return True

    def input_shape(self):
        return self.input.meta["val"].shape

    def weight_shape(self):
        weight_shape = self.weight.meta["val"].shape
        if self.weight_trans:
            return torch.Size([weight_shape[1], weight_shape[0]])
        return weight_shape

    def check_shape(self):
        if self.input is None:
            return False
        if self.weight is None:
            return False

        m1, k1 = self.input_shape()
        k2, n2 = self.weight_shape()
        if k1 != k2:
            logger.warning(
                f"MatMul pass: Unsupported dim input_shape: {self.input_shape()}, weight_shape: {self.weight_shape()}"
            )
            return False
        return True

    def set_bias(self, bias):
        if isinstance(bias, int):
            self.bias = bias
            return True
        if isinstance(bias, float):
            self.bias = bias
            return True

        m1, k1 = self.input_shape()
        k2, n2 = self.weight_shape()

        bias_shape = bias.meta["val"].shape
        if len(bias_shape) == 1:
            if (bias_shape != torch.Size([n2])) and (bias_shape != torch.Size([1])):
                return False
        elif len(bias_shape) == 2:
            m3, n3 = bias_shape
            if n2 != n3:
                return False
            if (m1 != m3) and (m3 != 1):
                return False
        self.bias = bias
        return True

    def set_act(self, act_str):
        if act_str in ["gelu", "relu", "silu", "sigmoid", "none"]:
            self.act = act_str
            return True
        return False


class DenseLayer(nn.Module):
    def forward(self, input, weight, weight_trans, bias, act):
        if weight_trans:
            weight = weight.transpose(0, 1)
        if bias is None:
            y = torch.matmul(input, weight)
        elif not isinstance(bias, torch.Tensor):
            y = torch.matmul(input, weight) + bias
        else:
            y = torch.addmm(bias, input, weight)
        if act == "gelu":
            y = F.gelu(y)
        elif act == "relu":
            y = F.relu(y)
        elif act == "silu":
            y = F.silu(y)
        elif act == "sigmoid":
            y = F.sigmoid(y)
        return y
