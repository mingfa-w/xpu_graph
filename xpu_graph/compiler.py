from typing import Callable, overload

import torch

from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import aot_module_simplified, aot_export_module

from .passes.pass_manager import PassManager
from .passes.patterns.pattern import Pattern
from .config import XpuGraphConfig, Target, OptLevel
from .utils import logger, setup_logger
import logging

class XpuGraph:
    call_time = 1

    def __init__(self, config: XpuGraphConfig = XpuGraphConfig()):
        self._config = config
        if self._config.debug:
            setup_logger(logging.DEBUG)
        else:
            setup_logger(logging.INFO)

        self._pass_manager = PassManager(self._config)


    def __call__(self, dynamo_gm, example_inputs, *args, **kwargs):

        # XpuGraph.call_time += 1
        # file_name = f"/data03/gaoyujia.01/zhangjihang/compile_{XpuGraph.call_time}"
        # with open(file_name, 'w') as file:
        #     file.write(f"{dynamo_gm.graph}")

        # # Unlift graph, which changes parameter node from "placehoder" to "get_attr"
        # # So we can do more aggresive constant folding
        # print(f"========= dynamo_gm {dynamo_gm.graph}")
        # file_name = f"/data03/gaoyujia.01/zhangjihang/dynamo_module.py"
        # with open(file_name, 'w') as file:
        #     file.write(f"{dynamo_gm.graph.python_code('dynamo_module')}")

        # # return dynamo_gm

        # logger.info("unlift graph start...")
        # lifted_gm, gs = aot_export_module(dynamo_gm, example_inputs, trace_joint=False)

        # print(f"========= lifted_gm {lifted_gm.graph}")
        # file_name = f"/data03/gaoyujia.01/zhangjihang/lifted_module.py"
        # with open(file_name, 'w') as file:
        #     file.write(f"{lifted_gm.graph.python_code('lifted_module')}")

        # # return lifted_gm

        # from xpu_graph.fx_utils import unlift_gm, unlift_gm_2_5
        # # unlifted_gm = unlift_gm(dynamo_gm, lifted_gm, gs)
        # unlifted_gm = unlift_gm_2_5(dynamo_gm, lifted_gm, gs)
        # logger.info("unlift graph complete")

        # print(f"========= unlifted_gm {unlifted_gm.graph}")
        # file_name = f"/data03/gaoyujia.01/zhangjihang/unlifted_module.py"
        # with open(file_name, 'w') as file:
        #     file.write(f"{unlifted_gm.graph.python_code('unlifted_module')}")

        # return unlifted_gm

        def compiler(gm, sample_inputs):
            # print(f"========= start {gm.graph}")
            # return gm
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

            print(f"========= finish {xpu_compiled.graph}")

            return xpu_compiled

        xpu_gm = aot_autograd(fw_compiler=compiler)(dynamo_gm, example_inputs)
        return xpu_gm

        return compiler(unlifted_gm, example_inputs)

    def get_pattern_manager(self):
        return self._pass_manager.get_pattern_manager()
