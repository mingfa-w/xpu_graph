from .compiler import XpuGraph, optimize_graph
from .config import Target, OptLevel, XpuGraphConfig
from .cache import XpuGraphCache, XpuGraphLocalCache, default_cache, no_cache
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

    assert "target" not in patch_configs, "target is not allowed to be set here"
    if "opt_level" in patch_configs and not isinstance(
        patch_configs["opt_level"], OptLevel
    ):
        patch_configs["opt_level"] = OptLevel(int(patch_configs["opt_level"]))
    config = dataclasses.replace(default_config, **patch_configs)

    if "cache" not in patch_configs:
        cache = no_cache() if is_training else default_cache()
    elif isinstance(patch_configs["cache"], str):
        cache = XpuGraphLocalCache(patch_configs["cache"])
    else:
        assert isinstance(
            patch_configs["cache"], XpuGraphCache
        ), "cache must be a XpuGraphCache instance"
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
