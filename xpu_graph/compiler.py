from typing import Callable, overload

import torch

from torch._dynamo import register_backend
from torch._functorch.aot_autograd import aot_module_simplified, aot_export_module

from .passes.pass_manager import PassManager
from .passes.patterns.pattern import Pattern
from .config import XpuGraphConfig, Target, OptLevel
from .utils import logger, setup_logger
import logging

class XpuGraph:
    def __init__(self, config: XpuGraphConfig = XpuGraphConfig()):
        self._config = config
        if self._config.debug:
            setup_logger(logging.DEBUG)
        else:
            setup_logger(logging.INFO)

        self._pass_manager = PassManager(self._config)


    def __call__(self, dynamo_gm, example_inputs, *args, **kwargs):
        # Unlift graph, which changes parameter node from "placehoder" to "get_attr"
        # So we can do more aggresive constant folding
        logger.info("unlift graph start...")
        lifted_gm, gs = aot_export_module(dynamo_gm, example_inputs, trace_joint=False)
        from xpu_graph.fx_utils import unlift_gm
        unlifted_gm = unlift_gm(dynamo_gm, lifted_gm, gs)
        logger.info("unlift graph complete")

        def compiler(gm, sample_inputs):
            from torch._guards import detect_fake_mode
            fake_mode = detect_fake_mode(sample_inputs)
            fake_inputs = [
                fake_mode.from_tensor(x) if isinstance(x, torch.Tensor) else x
                for x in sample_inputs
            ]
            fake_mode.allow_non_fake_inputs = True
            with fake_mode:
                logger.debug(f"before xpu_graph, graph like:\n {gm.graph}")
                logger.info("xpu_graph passes start...")

                xpu_compiled = self._pass_manager(gm, fake_inputs)

                logger.debug(f"after xpu_graph, graph like:\n {xpu_compiled.graph}")
                logger.info("xpu_graph passes complete")

                from xpu_graph.config import ExecuteMode
                if self._config.execute_mode == ExecuteMode.graph:
                    from .backends import make_graph
                    return make_graph(xpu_compiled, fake_inputs, self._config.target)

            return xpu_compiled

        return compiler(unlifted_gm, example_inputs)

    def get_pattern_manager(self):
        return self._pass_manager.get_pattern_manager()
