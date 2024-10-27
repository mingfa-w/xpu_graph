def get_all_patterns(target, opt_level) -> list:
    from xpu_graph.config import Target
    if target == Target.ascend:
        from .ascend.converter import get_all_patterns as get_all_converters
        from .ascend.optimizer import get_all_patterns as get_all_optimizers
        return get_all_converters(opt_level) + get_all_optimizers(opt_level)

    return []