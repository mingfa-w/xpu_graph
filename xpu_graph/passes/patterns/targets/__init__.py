from xpu_graph.config import Target

def get_all_patterns(config) -> list:
    if config.target == Target.mlu:
        from .mlu import get_all_patterns
        return get_all_patterns(config)
    return []
