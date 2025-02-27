import torch
import torch.utils._pytree as pytree

from torch.fx import map_arg
from torch.fx.experimental.proxy_tensor import make_fx, wrapper_and_args_for_make_fx
from torch.fx.proxy import Proxy, GraphAppendingTracer

from typing import Union, Callable


def unlift_gm(mod, gm, graph_signature):
    from torch.export.unflatten import _assign_attr, _AttrKind

    state_dict = {}
    for name, param in mod.named_parameters(remove_duplicate=False):
        state_dict[name] = param
        _assign_attr(
            param,
            gm,
            name,
            attr_kind=_AttrKind.PARAMETER,
        )
    for name, buffer in mod.named_buffers(remove_duplicate=False):
        state_dict[name] = buffer
        _assign_attr(
            buffer,
            gm,
            name,
            attr_kind=_AttrKind.BUFFER,
        )

    placeholder_nodes = [node for node in gm.graph.nodes if node.op == "placeholder"]
    lifted_inputs = []
    for node in placeholder_nodes:
        node_name = node.name
        if node_name in graph_signature.inputs_to_parameters:
            lifted_inputs.append(graph_signature.inputs_to_parameters[node_name])
        elif node_name in graph_signature.inputs_to_buffers:
            lifted_inputs.append(graph_signature.inputs_to_buffers[node_name])
        else:
            assert node_name in graph_signature.user_inputs
            lifted_inputs.append(None)

    from torch.export._unlift import _unlift

    outputs = list(gm.graph.nodes)[-1].args[0]
    mutated_outputs = []
    for out in outputs:
        if out in graph_signature.buffers_to_mutate:
            mutated_outputs.append(graph_signature.buffers_to_mutate[out.name])
        else:
            mutated_outputs.append(None)

    unlifted_gm = _unlift(
        gm,
        lifted_inputs,
        mutated_outputs,
        pytree.LeafSpec(),
        None,
        state_dict,
        {},
    )
    return unlifted_gm


def trace_and_inline(
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
            wrapped, record_module_stack=True, tracing_mode="fake", pre_dispatch=False
        )(f_arglist)
        traced.recompile()

        tracer = GraphAppendingTracer(graph_module.graph)
        p_arglist = list(map_arg(arglist, lambda arg: Proxy(arg, tracer)))
        rets = traced(p_arglist)
        return rets.node

    return inliner
