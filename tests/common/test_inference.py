import pytest

import torch
import torch.nn as nn
import xpu_graph
from xpu_graph import OptLevel
from xpu_graph.test_utils import is_similar, need_xpu_graph_logs, skip_xpu_graph_cache

from tests.common.test_models import all_models

device = "cpu"
data_type = torch.float32


def compare_inference(ModCls, backend, bsz=8, input_dim=16):
    golden = ModCls(input_dim).to(device=device, dtype=data_type)
    compiled = ModCls(input_dim).to(device=device, dtype=data_type)
    compiled.forward = torch.compile(compiled.forward, backend=backend, dynamic=False)
    compiled.load_state_dict(golden.state_dict())
    input = torch.randn((bsz, input_dim), device=device, dtype=data_type)
    target = torch.randn((bsz, 1), device=device, dtype=data_type)

    loss_fn = nn.MSELoss()

    with torch.no_grad():
        loss_golden = loss_fn(golden(input), target)
        loss_compiled = loss_fn(compiled(input), target)

    assert is_similar(loss_golden, loss_compiled)


class TestInference:
    def setup_class(self):
        infer_config = xpu_graph.XpuGraphConfig(
            is_training=False,
            opt_level=OptLevel.level2,
            freeze=False,
            debuggers=["inference"],
        )
        self.infer_backend = xpu_graph.XpuGraph(infer_config)

    @pytest.mark.parametrize(
        "ReproCls",
        all_models,
    )
    def test_inference(self, caplog, ReproCls):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.infer_backend):
            compare_inference(ReproCls, self.infer_backend)
            assert "Guarding" in caplog.text and "diverges" not in caplog.text


class TestFreezeInference:
    def setup_class(self):
        freeze_config = xpu_graph.XpuGraphConfig(
            is_training=False,
            opt_level=OptLevel.level2,
            freeze=True,
            debuggers=["inference"],
        )
        # Warning: DO NOT use create both freeze and non-freeze in the same test case,
        self.freeze_backend = xpu_graph.XpuGraph(freeze_config)

    @pytest.mark.parametrize(
        "ReproCls",
        all_models,
    )
    def test_freeze_inference(self, caplog, ReproCls):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.freeze_backend):
            compare_inference(ReproCls, self.freeze_backend)
            assert "Guarding" in caplog.text and "diverges" not in caplog.text


if __name__ == "__main__":

    config = xpu_graph.XpuGraphConfig(
        is_training=False,
        opt_level=OptLevel.level2,
        freeze=True,
        debug=True,
        debuggers=["inference"],
    )
    xpu_graph_backend = xpu_graph.XpuGraph(config)
    for ModCls in all_models:
        compare_inference(ModCls, xpu_graph_backend)

    config = xpu_graph.XpuGraphConfig(
        is_training=False,
        opt_level=OptLevel.level2,
        freeze=False,
        debug=True,
        debuggers=["inference"],
    )
    xpu_graph_backend = xpu_graph.XpuGraph(config)
    for ModCls in all_models:
        compare_inference(ModCls, xpu_graph_backend)
