import torch
import torch.utils._pytree as pytree
from torch.fx.node import map_arg, map_aggregate
from torch.fx.experimental.proxy_tensor import (
    make_fx,
    wrapper_and_args_for_make_fx,
    get_proxy_slot,
    set_original_aten_op,
)
from torch.fx.proxy import Proxy, GraphAppendingTracer

from typing import Union, Callable, Tuple, Optional, Dict
from enum import Enum


from torch._ops import OpOverload
from .utils import logger


class FxStage(Enum):
    inference = "inference"
    pregrad = "pregrad"
    forward = "forward"
    backward = "backward"


def trace_and_inline(
    predispatch: bool,
    graph_module: torch.fx.GraphModule,
    mod_or_func: Union[str, Callable],
):
    """Tracing a submodule or a function and inline the traced sub_graph"""

    if isinstance(mod_or_func, str):
        callable = graph_module.get_submodule(mod_or_func)
    else:
        callable = mod_or_func

    def inliner(*args, **kwargs):
        wrapped, arglist = wrapper_and_args_for_make_fx(callable, args, kwargs)

        # use the original (fake) tensor to avoid dynamic-control-flow issues
        f_arglist = list(map_arg(arglist, lambda arg: arg.meta["val"]))
        traced = make_fx(
            wrapped,
            pre_dispatch=predispatch,
            record_module_stack=True,
            tracing_mode="fake",
        )(f_arglist)
        traced.recompile()

        tracer = GraphAppendingTracer(graph_module.graph)
        p_arglist = list(map_arg(arglist, lambda arg: Proxy(arg, tracer)))
        rets = traced(p_arglist)
        return rets.node

    return inliner


from torch.fx.experimental.proxy_tensor import (
    _ModuleStackTracer,
    dispatch_trace,
    wrap_key,
    fake_signature,
    PreDispatchTorchFunctionMode,
    ProxyTorchDispatchMode,
    TorchFunctionMetadataMode,
    enable_python_dispatcher,
    disable_autocast_cache,
    _set_make_fx_tracer,
)
from contextlib import nullcontext, ExitStack


class _DecomposeTracer(_ModuleStackTracer):
    def call_module(self, m, forward, args, kwargs):
        return super().call_module(m, forward, args, kwargs)


class _FunctionalizeProxyTorchMode(ProxyTorchDispatchMode):
    def __init__(self, tracer, pre_dispatch):
        super().__init__(tracer, "fake", pre_dispatch)
        # Map the newest version back to the oldest version
        self.mem_snapshot = {}

    def get_snapeshot(self, obj):
        if isinstance(obj, torch.Tensor):
            return self.mem_snapshot.get(obj, obj)
        return obj

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: Tuple[torch._C._TensorMeta, ...],
        args: Tuple[object, ...] = (),
        kwargs: Optional[Dict[str, object]] = None,
    ):

        if func in inplace_decompositions:
            with set_original_aten_op(func):
                outplace_func = inplace_decompositions[func]
                orig = args[0]
                with self:
                    self.decomp_layers += 1
                    out = outplace_func(*args, **kwargs)
                    self.mem_snapshot[orig] = out
                    self.decomp_layers -= 1
                    return out

        if func._schema.is_mutable:
            logger.warn(
                f"Unsupported mutable op {func}. This may leads to incorrect results!. File a issue to us if you really needs this operator."
            )
        if func.is_view:
            args = (args[0], *map_aggregate(args[1:], self.get_snapeshot))
        else:
            args = map_aggregate(args, self.get_snapeshot)
        kwargs = map_aggregate(kwargs, self.get_snapeshot)
        return super().__torch_dispatch__(func, types, args, kwargs)


def decompose_fx(gm, is_training, *args):
    """
    A modified version of make_fx
    1. decompose graph ops into aten ir
    2. make in-place operations functionalized
    3. insert copy for mutations in the end of graph
    """
    fake_tensor_mode = torch._guards.detect_fake_mode(args)
    assert fake_tensor_mode is not None, "decompose_fx only supports fake inputs"

    # Generate placeholders for the inputs
    phs = pytree.tree_map(lambda _: torch.fx._symbolic_trace.PH, args)

    import inspect

    if (
        not hasattr(inspect.unwrap(gm), "__code__")
        or inspect.unwrap(gm).__code__.co_flags & inspect.CO_VARARGS
    ):
        # FX doesn't support varargs, so we gotta fake up a wrapper
        func = fake_signature(gm, len(phs))
    else:
        func = gm

    fx_tracer = _DecomposeTracer(gm)

    ## Setup modes
    #
    # We disable the autocast cache as the autocast cache causes type conversions on parameters to
    # check a cache, which introduces untracked tensors into the graph
    #
    # We also disable tracing by any other tensor proxy-based tracers except the current. The
    # purpose of `make_fx` is to produce graphmodules as a side effect; its internal execution is
    # thus irrelevant to any external functional trace.
    proxy_mode = _FunctionalizeProxyTorchMode(
        fx_tracer,
        pre_dispatch=is_training,
    )
    if is_training:
        proxy_function_mode = PreDispatchTorchFunctionMode(fx_tracer)
        # pre-autograd tracing uses per-dispatch-key modes,
        # which requires the python dispatcher
        python_dispatcher_mode = enable_python_dispatcher()
    else:
        proxy_function_mode = nullcontext()
        python_dispatcher_mode = nullcontext()

    torch_fn_metadata_mode = TorchFunctionMetadataMode(fx_tracer)

    with ExitStack() as stack:
        stack.enter_context(fake_tensor_mode)
        stack.enter_context(python_dispatcher_mode)
        stack.enter_context(proxy_function_mode)
        stack.enter_context(torch_fn_metadata_mode)
        stack.enter_context(proxy_mode)
        stack.enter_context(disable_autocast_cache())
        stack.enter_context(_set_make_fx_tracer(fx_tracer))

        t = dispatch_trace(
            wrap_key(func, args, fx_tracer, pre_dispatch=is_training),
            tracer=fx_tracer,
            concrete_args=tuple(phs),
        )
        output_node = t.graph.find_nodes(op="output")[0]
        mutated = {}
        with fx_tracer.graph.inserting_before(output_node):
            # Note: the iteration order of mem_ssa_versions matters
            for old_tensor, new_tensor in proxy_mode.mem_snapshot.items():
                old_node = get_proxy_slot(old_tensor, fx_tracer).proxy
                new_node = get_proxy_slot(new_tensor, fx_tracer).proxy
                from xpu_graph.passes.patterns.utils.check_ops import is_node_escaped

                if True:  # is_node_escaped(old_node.node):
                    # Note: mem_snapshot always maps a originally-existing tensor (inplace base) to a newly-created tensor (outplace result)
                    ret_node = fx_tracer.create_proxy(
                        "call_function",
                        aten.copy_.default,
                        args=(old_node, new_node),
                        kwargs={},
                    )
                    mutated[old_node.node] = ret_node.node
        output_node.args = (
            map_aggregate(output_node.args[0], lambda n: mutated.get(n, n)),
        )

    t.recompile()
    return t


inplace_decompositions = {}
from torch._decomp import _add_op_to_registry


def register_inplace_operator(inplace_op, outplace_op):
    _add_op_to_registry(inplace_decompositions, inplace_op, outplace_op)


aten = torch._ops.ops.aten
register_inplace_operator(aten.addbmm_, aten.addbmm)
register_inplace_operator(aten.addmm_, aten.addmm)
register_inplace_operator(aten.addmv_, aten.addmv)
register_inplace_operator(aten.baddbmm_, aten.baddbmm)
register_inplace_operator(aten.fill_, aten.fill)
register_inplace_operator(aten.gelu_, aten.gelu)
register_inplace_operator(aten.hardswish_, aten.hardswish)
register_inplace_operator(aten.hardtanh_, aten.hardtanh)
register_inplace_operator(aten.hardsigmoid_, aten.hardsigmoid)
register_inplace_operator(aten.__iand__, aten.__and__)
register_inplace_operator(aten.__ilshift__, aten.__lshift__)
register_inplace_operator(aten.index_put_, aten.index_put)
register_inplace_operator(aten.index_reduce_, aten.index_reduce)
register_inplace_operator(aten.__ior__, aten.__or__)
register_inplace_operator(aten.__irshift__, aten.__rshift__)
register_inplace_operator(aten.__ixor__, aten.__xor__)
register_inplace_operator(aten.leaky_relu_, aten.leaky_relu)
register_inplace_operator(aten.logit_, aten.logit)
register_inplace_operator(aten.relu_, aten.relu)
register_inplace_operator(aten.renorm_, aten.renorm)
register_inplace_operator(aten.round_, aten.round)
register_inplace_operator(aten.scatter_, aten.scatter)
register_inplace_operator(aten.scatter_add_, aten.scatter_add)
register_inplace_operator(aten.scatter_reduce_, aten.scatter_reduce)
register_inplace_operator(aten.silu_, aten.silu)
register_inplace_operator(aten.add_, aten.add)
register_inplace_operator(aten.bitwise_and_, aten.bitwise_and)
register_inplace_operator(aten.bitwise_left_shift_, aten.bitwise_left_shift)
register_inplace_operator(aten.bitwise_not_, aten.bitwise_not)
register_inplace_operator(aten.bitwise_or_, aten.bitwise_or)
register_inplace_operator(aten.bitwise_right_shift_, aten.bitwise_right_shift)
register_inplace_operator(aten.bitwise_xor_, aten.bitwise_xor)
register_inplace_operator(aten.mul_, aten.mul)
register_inplace_operator(aten.div_, aten.div)
register_inplace_operator(aten.logical_and_, aten.logical_and)
register_inplace_operator(aten.logical_not_, aten.logical_not)
register_inplace_operator(aten.logical_or_, aten.logical_or)
register_inplace_operator(aten.logical_xor_, aten.logical_xor)
register_inplace_operator(aten.sub_, aten.sub)
register_inplace_operator(aten.sigmoid_, aten.sigmoid)
register_inplace_operator(aten.zero_, aten.zeros_like)
register_inplace_operator(aten.copy_, aten.copy)
register_inplace_operator(aten.detach_, aten.detach)
