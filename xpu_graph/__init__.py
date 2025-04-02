from .compiler import XpuGraph, optimize_graph
from .config import Target, OptLevel, XpuGraphConfig
from .cache import XpuGraphCache, default_cache, no_cache
import dataclasses

__all__ = [
    "XpuGraph",
    "XpuGraphConfig",
    "Target",
    "OptLevel",
    "optimize_graph",
    "XpuGraphCache",
    "default_cache",
    "mlu_compiler",
]


def mlu_compiler(
    is_training: bool,
    **patch_configs,
) -> XpuGraph:
    """
    Create an MLU compiler configuration and return an XpuGraph instance.

    Possible Patch Args:
        freeze: Whether to freeze the graph.
        opt_level: Optimization level.
        constant_folding: Whether to enable constant folding.
        cache: Cache for compiled graphs. Uses default cache if None.
        debug: Whether to enable debug mode.
        vendor_compiler_config: Additional vendor-specific compiler configuration.

    Returns:
        An XpuGraph instance configured for MLU.
    """

    default_config = _MLU_TRAIN_CONFIG if is_training else _MLU_INFER_CONFIG
    config = dataclasses.replace(default_config, **patch_configs)
    if "cache" not in patch_configs:
        cache = default_cache() if is_training else no_cache()
    else:
        cache = patch_configs["cache"]
    if not is_training:
        import torch_mlu_ops
    return XpuGraph(config, cache)


_MLU_TRAIN_CONFIG = XpuGraphConfig(
    is_training=True,
    debug=False,
    target=Target.mlu,
    enable_cache=True,
    freeze=False,
    opt_level=OptLevel.level2,
    constant_folding=False,
    vendor_compiler_config={"mode": "reduce-overhead"},
)

_MLU_INFER_CONFIG = XpuGraphConfig(
    is_training=False,
    debug=False,
    target=Target.mlu,
    enable_cache=True,
    freeze=False,
    opt_level=OptLevel.level2,
    constant_folding=False,
    vendor_compiler_config={"mode": "reduce-overhead"},
)
