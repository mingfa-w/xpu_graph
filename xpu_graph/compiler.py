from typing import Callable, overload

import torch

from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import aot_export_module
from torch.fx.experimental.proxy_tensor import make_fx
from torch._guards import detect_fake_mode

from .passes.pass_manager import PassManager
from .passes.patterns.pattern import Pattern
from .config import XpuGraphConfig, Target, OptLevel
from .utils import logger, setup_logger
from .cache import XpuGraphCache, default_cache
import logging
from .fx_utils import FxStage
from functools import partial


class XpuGraph:
    def __init__(
        self,
        config: XpuGraphConfig = XpuGraphConfig(),
        cache: XpuGraphCache = default_cache(),
    ):
        self._config = config
        if self._config.debug:
            setup_logger(logging.DEBUG)
        else:
            setup_logger(logging.INFO)
        if self._config.freeze:
            # The configuration in this inductor affects the return value of is_parameter_freezing(),
            # thereby influencing the process of generating the fx_graph in dynamo. The current code
            # in the community is not very clean, and it would be more reasonable to place this
            # configuration under dynamo. You can refer to this link for more information.
            # https://github.com/pytorch/pytorch/blob/release/2.5/torch/_dynamo/utils.py#L3061
            torch._inductor.config.freezing = True

        self._pass_manager = PassManager(self._config)
        self._cache = cache

    def __call__(self, dynamo_gm, example_inputs, *args, **kwargs):
        # Every invoke of inner compile will produce a new graph; use id to fetch cached graphs
        graph_id = 0

        fake_mode = detect_fake_mode(example_inputs)
        fake_inputs = [
            fake_mode.from_tensor(x) if isinstance(x, torch.Tensor) else x
            for x in example_inputs
        ]
        fake_mode.allow_non_fake_inputs = True

        hashkey = self._cache.cache_key(dynamo_gm, fake_inputs, self._config)

        def _compiler(gm, fake_inputs, stage: FxStage):
            nonlocal graph_id
            graph_id += 1
            logger.debug(f"Stage: {stage}, graph id: {graph_id}")

            with fake_mode:
                logger.debug(f"before xpu_graph, graph like:\n {gm.graph}")
                logger.info(f"before xpu_graph, nodes num: {len(gm.graph.nodes)}")
                logger.info("xpu_graph passes start...")

                xpu_compiled = self._cache.load_gm(hashkey, graph_id)
                if xpu_compiled is None:
                    xpu_compiled = self._pass_manager(gm, fake_inputs, stage)
                    xpu_compiled = self._cache.save_gm(hashkey, graph_id, xpu_compiled)

                logger.debug(f"after xpu_graph, graph like:\n {xpu_compiled.graph}")
                logger.info("xpu_graph passes complete")
                logger.info(
                    f"after xpu_graph, nodes num: {len(xpu_compiled.graph.nodes)}"
                )

                if stage == FxStage.pregrad:
                    return xpu_compiled

                if self._config.vendor_compiler_config:

                    from .backends import vendor_compiler

                    return vendor_compiler(
                        xpu_compiled,
                        fake_inputs,
                        self._config.target,
                        self._config.vendor_compiler_config,
                    )

            return xpu_compiled

        logger.debug(f"before dispatch: graph like:\n {dynamo_gm.graph}")
        logger.info("dispatch graph start...")
        with fake_mode:
            dispatched_gm = make_fx(
                dynamo_gm,
                tracing_mode="fake",
                pre_dispatch=True,
                record_module_stack=True,
            )(*fake_inputs)
        logger.debug(f"after dispatch, graph like:\n {dispatched_gm.graph}")

        if self._config.freeze:
            logger.info("unlift graph start...")
            lifted_gm, gs = aot_export_module(
                dispatched_gm, fake_inputs, trace_joint=False
            )
            logger.debug(f"before unlift, graph like:\n {lifted_gm.graph}")
            logger.debug(f"graph signature: {gs}")
            from xpu_graph.fx_utils import unlift_gm

            unlifted_gm = unlift_gm(dispatched_gm, lifted_gm, gs)
            logger.info("unlift graph complete")
            logger.debug(f"after unlift, graph like:\n {unlifted_gm.graph}")

            return _compiler(unlifted_gm, fake_inputs, stage=FxStage.forward)
        else:
            # Since: 1. dynamo has eliminated control-flow for input GraphModule
            #    and 2. aot_autograd traces grad again
            # It's okay use optimized infer-graph for training as well
            pregrad_gm = _compiler(dispatched_gm, fake_inputs, stage=FxStage.pregrad)
            xpu_gm = aot_autograd(
                fw_compiler=partial(_compiler, stage=FxStage.forward),
                bw_compiler=partial(_compiler, stage=FxStage.backward),
            )(pregrad_gm, fake_inputs)
            return xpu_gm

    def get_pattern_manager(self):
        return self._pass_manager.get_pattern_manager()
