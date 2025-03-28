import torch
from torch._inductor.pattern_matcher import PatternMatcherPass, register_replacement, fwd_only, Match

def is_valid_sum_bool(match: Match):
    input = match.kwargs['x0'].meta.get('val')
    if (input.dtype != torch.bool):
        return False
    if (match.output_node().meta.get('val').dtype != torch.int64):
        return False
    # We convert int64 to int32 only when it fits in int32 ranges
    if (input.numel() > 2**31 - 1):
        return False
    return True

def sum_bool_replacement(x0):
    return torch.sum(x0, dtype=torch.int32)

# FIXME: The target pattern is only sum.dim_IntList?
def sum_bool_pattern(x0):
    return torch.ops.aten.sum.dim_IntList(x0, None)

custom_patterns = PatternMatcherPass()

def _register_pattern_once():
    if hasattr(_register_pattern_once, "registered"):
        return
    x = torch.empty((86, 128)).bool().npu()
    register_replacement(sum_bool_pattern, 
                        sum_bool_replacement,
                        [x],
                        fwd_only,
                        [custom_patterns],
                        extra_check=is_valid_sum_bool,
                        )
    setattr(_register_pattern_once, "registered", 1)

def sum_bool_pass(graph):
    _register_pattern_once()
    custom_patterns.apply(graph)