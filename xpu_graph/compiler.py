from typing import Callable, overload

import torch

from torch._dynamo.backends.common import aot_autograd
from torch._functorch.aot_autograd import aot_export_module

from .passes.pass_manager import PassManager
from .passes.patterns.pattern import Pattern
from .config import XpuGraphConfig, Target, OptLevel
from .utils import logger, setup_logger
import logging

from torch._dynamo.convert_frame import compile_lock
from torch.utils._python_dispatch import _disable_current_modes
import os
import hashlib
import pickle


class XpuGraph:
    def __init__(
        self, config: XpuGraphConfig = XpuGraphConfig(), cache_path: os.PathLike = None
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
        if cache_path is not None:
            cache_path = os.path.abspath(cache_path)
            os.makedirs(cache_path, exist_ok=True)
        self._cache_path = cache_path

    def __call__(self, dynamo_gm, example_inputs, *args, **kwargs):
        def _compiler(gm, sample_inputs):
            from torch._guards import detect_fake_mode

            fake_mode = detect_fake_mode(sample_inputs)
            fake_inputs = [
                fake_mode.from_tensor(x) if isinstance(x, torch.Tensor) else x
                for x in sample_inputs
            ]
            fake_mode.allow_non_fake_inputs = True

            if self._cache_path is not None:
                hashkey = hashlib.md5(
                    f"{gm}-{fake_inputs}-{self._config}".encode()
                ).hexdigest()
                fname = f"xpu_graph_{hashkey}.pt"
                artifact_cache = os.path.join(self._cache_path, fname)
            else:
                artifact_cache = None

            with fake_mode:
                logger.debug(f"before xpu_graph, graph like:\n {gm.graph}")
                logger.info(f"before xpu_graph, nodes num: {len(gm.graph.nodes)}")
                logger.info("xpu_graph passes start...")

                if artifact_cache is None or not os.path.exists(artifact_cache):
                    xpu_compiled = self._pass_manager(gm, fake_inputs)

                    if artifact_cache is not None:
                        logger.info(f"Save cache in location: {artifact_cache}")
                        with compile_lock, _disable_current_modes():
                            with open(artifact_cache, "wb+") as f:
                                pickle.dump(xpu_compiled, f)

                if artifact_cache is not None:
                    # Alwayrs use the deserialized version to keep consistence for inductor
                    logger.info(f"Use cache in location: {artifact_cache}")
                    with compile_lock, _disable_current_modes():
                        with open(artifact_cache, "rb") as f:
                            xpu_compiled = pickle.load(f)

                logger.debug(f"after xpu_graph, graph like:\n {xpu_compiled.graph}")
                logger.info("xpu_graph passes complete")
                logger.info(
                    f"after xpu_graph, nodes num: {len(xpu_compiled.graph.nodes)}"
                )

                if self._config.vendor_compiler:
                    if self._cache_path is not None:
                        os.environ["TORCHINDUCTOR_CACHE_DIR"] = os.path.join(
                            self._cache_path, "inductor"
                        )
                    from .backends import vendor_compiler

                    return vendor_compiler(
                        xpu_compiled,
                        fake_inputs,
                        self._config.target,
                        self._config.vendor_compiler,
                    )

            return xpu_compiled

        if self._config.freeze:
            logger.info("unlift graph start...")
            lifted_gm, gs = aot_export_module(
                dynamo_gm, example_inputs, trace_joint=False
            )
            print(f"dynamo_gm: {dynamo_gm.graph}")
            print(f"graph signature: {gs}")

            logger.debug(f"before unlift, graph like:\n {lifted_gm.graph}")

            from xpu_graph.fx_utils import unlift_gm

            unlifted_gm = unlift_gm(dynamo_gm, lifted_gm, gs)
            logger.info("unlift graph complete")
            logger.debug(f"after unlift, graph like:\n {unlifted_gm.graph}")

            return _compiler(unlifted_gm, example_inputs)
        else:
            xpu_gm = aot_autograd(fw_compiler=_compiler)(dynamo_gm, example_inputs)
            return xpu_gm

    def get_pattern_manager(self):
        return self._pass_manager.get_pattern_manager()
