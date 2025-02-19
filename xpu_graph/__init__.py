from .compiler import XpuGraph
from .config import Target, OptLevel, XpuGraphConfig
from .cache import XpuGraphCache, default_cache

__all__ = ["XpuGraph", "XpuGraphConfig", "Target", "OptLevel"]


def mlu_compiler(
    freeze: bool = True,
    opt_level: OptLevel = OptLevel.level1,
    graph: bool = True,
    constant_folding: bool = False,
    cache: XpuGraphCache = default_cache(),
):
    config = XpuGraphConfig(
        target=Target.mlu,
        freeze=freeze,
        opt_level=opt_level,
        constant_folding=constant_folding,
    )
    if graph:
        config.vendor_compiler = {"mode": "reduce-overhead"}
    return XpuGraph(config, cache)
