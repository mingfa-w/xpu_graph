import torch
from xpu_graph.config import Target
from xpu_graph.utils import logger
from typing import Callable

def make_graph(gm: torch.fx.GraphModule, fake_inputs: list, target: Target) -> Callable:

    if target == Target.ascend:
        from .ascend import npu_compile
        logger.info("npu_compile start...")
        npu_compiled = npu_compile(gm, fake_inputs)
        logger.info("npu_compile complete")
        return npu_compiled
    elif target == Target.mlu:
        from .mlu import mlu_compile
        logger.info("mlu_compile start...")
        mlu_compiled = mlu_compile(gm, fake_inputs)
        logger.info("mlu_compile complete")
        return mlu_compiled

    return gm

