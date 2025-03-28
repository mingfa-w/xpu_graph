import torch
import torch.nn as nn
import torch.nn.functional as F
from xpu_graph.config import OptLevel
import xpu_graph
from xpu_graph.test_utils import (
    is_similar,
    maybe_similar,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)
import pytest

device = "cpu"
data_type = torch.float32
aten = torch.ops.aten


def fn0(inputs, weight, bias):
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(
        inputs, dim=-1, keepdim=True, unbiased=False
    )  # unbiased=False == tf.nn.moments
    normalized = (inputs - mean) / ((variance + 1e-6) ** (0.5))
    outputs = weight * normalized + bias
    return outputs


def fn1(inputs, weight, bias):
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(
        inputs, dim=-1, keepdim=True, unbiased=False
    )  # unbiased=False == tf.nn.moments
    normalized = (inputs - mean) * torch.rsqrt(1e-6 + variance)
    outputs = bias + normalized
    return outputs


def fn2(inputs, weight, bias):
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, correction=0)
    normalized = (inputs - mean) / torch.sqrt(variance + 1e-5)
    return normalized


def fn3(inputs, weight, bias):
    mean = torch.mean(inputs, dim=-1, keepdim=True)
    variance = torch.var(inputs, dim=-1, keepdim=True, correction=0)
    normalized = (inputs - mean) * ((variance + 1e-5) ** (-0.5))
    return weight * normalized


def layernorm_test(xpu_graph, func):
    inputs = torch.randn((8, 1024), device=device, dtype=data_type)
    weight = torch.randn((1024), device=device, dtype=data_type)
    bias = None
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)
    if func == fn0 or func == fn1:
        bias = torch.randn((1024,), device=device, dtype=data_type)
        norm = compiled(inputs, weight, bias)
        norm1 = func(inputs, weight, bias)
        assert is_similar(norm1, norm)
    if func == fn2 or func == fn3:
        norm = compiled(inputs, weight, bias)
        norm1 = func(inputs, weight, bias)
        assert is_similar(norm1, norm)


def layernorm_test_with_loss_and_grad(xpu_graph, func):
    inputs = torch.randn(
        (
            8,
            1024,
        ),
        device=device,
        dtype=data_type,
    )
    weight = torch.randn((1024,), device=device, dtype=data_type)
    bias = torch.randn((1024,), device=device, dtype=data_type)
    ref = torch.randn((8, 1024), device=device, dtype=data_type)
    compiled = torch.compile(func, backend=xpu_graph, dynamic=False)

    inputs0, weight0, bias0 = (
        inputs.clone().requires_grad_(),
        weight.clone().requires_grad_(),
        bias.clone().requires_grad_(),
    )
    norm0 = compiled(inputs0, weight0, bias0)
    loss0 = F.mse_loss(norm0, ref)
    loss0.backward()

    inputs1, weight1, bias1 = (
        inputs.clone().requires_grad_(),
        weight.clone().requires_grad_(),
        bias.clone().requires_grad_(),
    )
    norm1 = func(inputs1, weight1, bias1)
    loss1 = F.mse_loss(norm1, ref)
    loss1.backward()

    assert is_similar(norm0.detach(), norm1.detach())
    assert is_similar(loss0.detach(), loss1.detach())

    assert is_similar(inputs0.grad, inputs1.grad)

    assert maybe_similar(weight0.grad, weight1.grad)
    assert maybe_similar(bias0.grad, bias1.grad)


class TestLayerNorm:
    def setup_class(self):
        infer_config = xpu_graph.XpuGraphConfig(
            is_training=False, opt_level=OptLevel.level2
        )
        self.infer_backend = xpu_graph.XpuGraph(infer_config)
        train_config = xpu_graph.XpuGraphConfig(
            is_training=True, opt_level=OptLevel.level2
        )
        self.train_backend = xpu_graph.XpuGraph(train_config)

    @pytest.mark.parametrize(
        "pattern_func",
        [fn0, fn1, fn2, fn3],
    )
    def test_layernrom_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.infer_backend):
            layernorm_test(self.infer_backend, pattern_func)
        assert "Pattern.FusedLayerNorm changed graph" in caplog.text

    @pytest.mark.parametrize(
        "pattern_func",
        [fn2, fn3],
    )
    def test_layernrom_patterns_with_loss_and_grad(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.train_backend):
            layernorm_test_with_loss_and_grad(self.train_backend, pattern_func)
        assert "Pattern.FusedLayerNorm changed graph" in caplog.text


if __name__ == "__main__":
    infer_config = xpu_graph.XpuGraphConfig(
        is_training=False, opt_level=OptLevel.level2, debug=True
    )
    infer_backend = xpu_graph.XpuGraph(infer_config)
    layernorm_test(infer_backend, fn0)
    layernorm_test(infer_backend, fn1)
    layernorm_test(infer_backend, fn2)
    layernorm_test(infer_backend, fn3)
    train_config = xpu_graph.XpuGraphConfig(
        is_training=True, opt_level=OptLevel.level2, debug=True
    )
    train_backend = xpu_graph.XpuGraph(train_config)
    layernorm_test_with_loss_and_grad(train_backend, fn0)
    layernorm_test_with_loss_and_grad(train_backend, fn1)
    layernorm_test_with_loss_and_grad(train_backend, fn2)
    layernorm_test_with_loss_and_grad(train_backend, fn3)
