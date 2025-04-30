import pytest

import torch
import torch_mlu
import xpu_graph

from xpu_graph.test_utils import is_similar

import torch_mlu_ops as ops


aten = torch.ops.aten
device = "mlu:0"
data_type = torch.float32

act_mode_dict = {
    "relu": torch.nn.functional.relu,
    "gelu": torch.nn.functional.gelu,
    "silu": torch.nn.functional.silu,
}
from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    assertTensorsEqual,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)


def fn0(ffn_input, ffn_weight1, ffn_weight2, bias1=None, bias2=None, act="none"):
    output = torch.matmul(ffn_input, ffn_weight1)
    if bias1 is not None:
        output = output + bias1
    if act != "none":
        torch_act = act_mode_dict[act]
        output = torch_act(output)
    output = torch.matmul(output, ffn_weight2)
    if bias2 is not None:
        output = output + bias2
    return output


def fn1(ffn_input, ffn_weight1, ffn_weight2, bias1=None, bias2=None, act="silu"):
    output = fn0(ffn_input, ffn_weight1, ffn_weight2, bias1, bias2, act)
    return output


def fn2(ffn_input, ffn_weight1, ffn_weight2, bias1=None, bias2=None, act="silu"):
    output = fn0(ffn_input, ffn_weight1, ffn_weight2, bias1, bias2, act)
    return output


def fn3(ffn_input, ffn_weight1, ffn_weight2, bias1=None, bias2=None, act="silu"):
    output = torch.matmul(ffn_input, ffn_weight1.transpose(1, 0))
    if bias1 is not None:
        output = output + bias1
    torch_act = act_mode_dict[act]
    output = torch_act(output)
    output = torch.matmul(output, ffn_weight2.transpose(1, 0))
    if bias2 is not None:
        output = output + bias2
    return output.view(-1)


def ffn_test(xpu_graph_backend, func):
    with torch.no_grad():
        batch = 12800
        input_size = 256
        weight_size1 = 128
        weight_size2 = 32
        dtype = torch.bfloat16

        tinyffn_input = torch.randn(
            batch, input_size, dtype=dtype, device=device
        )
        weight1 = torch.randn(
            input_size, weight_size1, dtype=dtype, device=device
        )
        weight2 = torch.randn(
            weight_size1, weight_size2, dtype=dtype, device=device
        )

        bias1 = torch.randn(weight_size1, dtype=dtype, device=device)
        bias2 = torch.randn(weight_size2, dtype=dtype, device=device)
        act_mode = "none"
        if func in [fn0]:
            #bias1 = None
            bias2 = None
            act_mode = "relu"

        args = [tinyffn_input]
        args += [
            weight1,
            weight2,
            bias1,
            bias2,
        ]
        args += [act_mode]

        compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
        res1 = func(*args)
        res = compiled(*args)

        assertTensorsEqual(
            res1.cpu().float(), res.cpu().float(), 0.005, use_MSE=True, use_RAE=True
        )


class TestFFN:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False, freeze=False, opt_level=OptLevel.level2
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0],
        #[fn0, fn1, fn2, fn3],
    )
    def test_ffn_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            ffn_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.FusedTinyFFN changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False, freeze=False, opt_level=OptLevel.level2
    )
    ffn_test(xpu_graph_backend, fn0)
    #ffn_test(xpu_graph_backend, fn1)
    #ffn_test(xpu_graph_backend, fn2)
    #ffn_test(xpu_graph_backend, fn3)
