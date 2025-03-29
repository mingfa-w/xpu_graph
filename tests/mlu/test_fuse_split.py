import pytest

import torch
import torch_mlu
import xpu_graph
from xpu_graph import OptLevel
from xpu_graph.test_utils import (
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)


device = "mlu:0"
aten = torch.ops.aten


def fn0(x):
    parts = torch.split(x, 380, dim=1)
    return parts[0] + parts[1] + parts[2] + parts[3]

def split_test(xpu_graph_backend, func):
    for batch in (2048, 51):
        a = torch.randn(batch, 380 * 4, requires_grad=True, device=device)
        a_clone = a.detach().clone().requires_grad_()
        compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
        res_ref = func(a)
        res = compiled(a_clone)
        print(res.shape)
        print(res_ref.shape)
        #assert torch.equal(res.cpu().float(), res_ref.cpu().float())

        loss = res_ref.sum()
        loss.backward()
        grad = a.grad

        loss_ref = res.sum()
        loss_ref.backward()
        grad_ref = a_clone.grad

        assert torch.equal(grad.cpu(), grad_ref.cpu())

class TestSplit:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(
            is_training=False, freeze=True, opt_level=OptLevel.level1
        )

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
        ],
    )
    def test_split_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            split_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.FusedCatSlice changed graph" in caplog.text



if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=True, freeze=True, opt_level=OptLevel.level1, debug=True, vendor_compiler_config=None
    )
    split_test(xpu_graph_backend, fn0)
