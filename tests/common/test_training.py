import pytest

import torch
import torch.nn as nn
import xpu_graph
from xpu_graph import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache

from tests.common.test_models import all_models

device = "cpu"
data_type = torch.float32


def compare_training(ModCls, backend, nsteps=10, bsz=8, input_dim=16):
    golden = ModCls(input_dim).to(device=device, dtype=data_type)
    compiled = ModCls(input_dim).to(device=device, dtype=data_type)
    compiled.forward = torch.compile(compiled.forward, backend=backend, dynamic=False)
    compiled.load_state_dict(golden.state_dict())
    input = torch.randn((bsz, input_dim), device=device, dtype=data_type)
    target = torch.randn((bsz, 1), device=device, dtype=data_type)
    optimizer_golden = torch.optim.AdamW(golden.parameters())
    optimizer_compiled = torch.optim.AdamW(compiled.parameters())
    optimizer_compiled.load_state_dict(optimizer_golden.state_dict())
    print(optimizer_golden.state_dict(), optimizer_compiled.state_dict())

    loss_fn = nn.MSELoss()

    for i in range(nsteps):

        optimizer_golden.zero_grad()
        loss_golden = loss_fn(golden(input), target)
        loss_golden.backward()
        optimizer_golden.step()

        optimizer_compiled.zero_grad()
        loss_compiled = loss_fn(compiled(input), target)
        loss_compiled.backward()
        optimizer_compiled.step()

        print(f"Step: {i} golden: {loss_golden}, compiled: {loss_compiled}")
        assert is_similar(loss_golden, loss_compiled)
        for p_name, p_golden in golden.named_parameters():
            p_compiled = compiled.get_parameter(p_name)
            assert is_similar(p_golden, p_compiled)


class TestTraining:
    def setup_class(self):
        train_config = xpu_graph.XpuGraphConfig(
            is_training=True,
            opt_level=OptLevel.level2,
            freeze=False,
            debuggers=["autograd", "function"],
        )
        self.train_backend = xpu_graph.XpuGraph(train_config)

    @pytest.mark.parametrize(
        "ReproCls",
        all_models,
    )
    def test_layernorm_patterns_with_loss_and_grad(self, caplog, ReproCls):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.train_backend):
            compare_training(ReproCls, self.train_backend)
            assert "diverges" not in caplog.text


if __name__ == "__main__":
    config = xpu_graph.XpuGraphConfig(
        is_training=True,
        opt_level=OptLevel.level2,
        freeze=False,
        debug=True,
        debuggers=["autograd", "function"],
    )
    xpu_graph_backend = xpu_graph.XpuGraph(config)
    for ModCls in all_models:
        compare_training(ModCls, xpu_graph_backend)
