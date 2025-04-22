import pytest
import math

import torch
import torch_mlu
import xpu_graph

from xpu_graph.config import OptLevel
from xpu_graph.test_utils import (
    assertTensorsEqual,
    need_xpu_graph_logs,
    skip_xpu_graph_cache,
)

def get_profiler(out_path):
    profiler = torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.MLU,
        ],
        schedule=torch.profiler.schedule(wait=2, warmup=0, active=1, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=torch_mlu.profiler.tensorboard_trace_handler(out_path),
    )
    return profiler

def fn0(a, b):
    slice_ = a[:, :, 32:33]
    mul = torch.mul(slice_, b)
    squeeze = mul.squeeze(dim=[2])
    out = torch.sum(squeeze, dim=[1], keepdim=True)
    return out

def mul_sum_test(xpu_graph_backend, func):
    batch = 86
    dtype = torch.half
    a = torch.randn(batch, 1392, 33, dtype=dtype, device="mlu")
    b = torch.randn(1, 1392, 1, dtype=dtype, device="mlu")

    res1 = func(a, b)
    compiled = torch.compile(func, backend=xpu_graph_backend, dynamic=False)
    res = compiled(a, b)

    #from datetime import datetime
    #import time
    #import os
    #prof_round = 3
    #current_time = datetime.now()
    #timestamp_str = current_time.strftime("%Y%m%d_%H%M%S")
    #root_profiling_dir = "mulsum"
    #os.makedirs(root_profiling_dir, exist_ok=True)
    #with get_profiler(f"{root_profiling_dir}/profile_{timestamp_str}") as prof:
    #    while prof_round:
    #        res = compiled(a, b)
    #        prof_round = prof_round - 1
    #        prof.step()
    #    torch.mlu.synchronize()
    #print(prof.key_averages().table(sort_by="device_time"))

    assertTensorsEqual(
        res1.cpu().float(), res.cpu().float(), 0.002, use_MSE=True, use_RAE=True
    )

class TestMulSum:
    def setup_class(self):
        self.xpu_graph_backend = xpu_graph.mlu_compiler(is_training=False)

    @pytest.mark.parametrize(
        "pattern_func",
        [
            fn0,
        ],
    )
    def test_mul_sum_patterns(self, caplog, pattern_func):
        with need_xpu_graph_logs(), skip_xpu_graph_cache(self.xpu_graph_backend):
            mul_sum_test(self.xpu_graph_backend, pattern_func)
        assert "Pattern.FusedMulSum changed graph" in caplog.text


if __name__ == "__main__":
    xpu_graph_backend = xpu_graph.mlu_compiler(
        is_training=False,
    )
    mul_sum_test(xpu_graph_backend, fn0)
