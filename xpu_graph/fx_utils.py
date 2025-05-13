import itertools
from typing import Union, Callable
from enum import Enum
from unittest.mock import patch
from contextlib import nullcontext

import torch
import torch.utils._pytree as pytree
import torch.fx as fx
from torch.export.unflatten import _assign_attr, _AttrKind
from torch.fx import map_arg
from torch.fx.experimental.proxy_tensor import make_fx, wrapper_and_args_for_make_fx
from torch.fx.proxy import Proxy, GraphAppendingTracer
from torch._guards import detect_fake_mode
from torch._dispatch.python import enable_python_dispatcher

from torch._decomp.decompositions_for_rng import PhiloxStateTracker
from torch._dynamo.utils import preserve_rng_state

from torch._functorch.aot_autograd import (
    AOTConfig,
    create_functional_call,
)
from torch._functorch._aot_autograd.collect_metadata_analysis import (  # noqa: F401
    run_functionalized_fw_and_collect_metadata,
)
from torch._functorch._aot_autograd.dispatch_and_compile_graph import (
    aot_dispatch_autograd_graph,
    aot_dispatch_base_graph,
)


FX_COUNT = itertools.count()


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
        # and "real" tracing mode should be used to avoid fakify again
        f_arglist = list(map_arg(arglist, lambda arg: arg.meta["val"]))
        traced = make_fx(
            wrapped,
            pre_dispatch=predispatch,
            record_module_stack=True,
            tracing_mode="real",
        )(f_arglist)
        traced.recompile()

        tracer = GraphAppendingTracer(graph_module.graph)
        p_arglist = list(map_arg(arglist, lambda arg: Proxy(arg, tracer)))
        rets = traced(p_arglist)
        return rets.node

    return inliner


def fakify_tensors(full_args):
    fake_mode = detect_fake_mode(full_args)
    fake_flat_args = [
        fake_mode.from_tensor(x) if isinstance(x, torch.Tensor) else x
        for x in full_args
    ]
    return fake_mode, fake_flat_args


def dispatch_graph(gm, example_inputs, *, stage, decompositions=None):
    params_flat, params_spec, full_args, aot_config = _collect_params_and_inputs_info(
        gm, example_inputs
    )

    params_len = len(params_flat)

    # Use the config similar to aot_export_module
    aot_config.is_export = True
    aot_config.pre_dispatch = stage == FxStage.pregrad
    aot_config.no_tangents = True
    if decompositions is not None:
        aot_config.decompositions = decompositions

    flat_fn = create_functional_call(gm, params_spec, params_len)

    ctx = nullcontext if stage == FxStage.pregrad else torch.no_grad
    with ctx():
        fake_mode, fake_flat_args = fakify_tensors(full_args)
        shape_env = fake_mode.shape_env

        dispatched_gm, fw_metadata = _invoke_dispatcher(
            flat_fn, fake_flat_args, fake_mode, shape_env, aot_config, stage
        )

    num_tokens = len(fw_metadata.tokens)
    assert num_tokens == 0

    input_nodes = _unlift_params(gm, dispatched_gm)
    _insert_mutations(dispatched_gm, fw_metadata, input_nodes)
    dispatched_gm.graph.lint()
    dispatched_gm.recompile()

    fake_inputs = fake_flat_args[params_len:]
    return dispatched_gm, fake_inputs


def find_nodes(graph, *, op: str, target=None):
    result = []
    for node in graph.nodes:
        if node.op == op and (target is None or node.target == target):
            result.append(node)
    return result


def _collect_params_and_inputs_info(gm, example_inputs):
    # Copied from torch._functorch.aot_autograd.
    # lift gm paramters and buffers, and construct basic aot_config with collected infos
    params = {
        **dict(gm.named_parameters(remove_duplicate=False)),
        **dict(gm.named_buffers(remove_duplicate=False)),
    }
    params_flat, params_spec = pytree.tree_flatten(params)
    params_flat = list(params_flat)

    seen_sources = set()

    full_args = []
    # First, the params
    full_args.extend(params_flat)

    if tracing_context := torch._guards.TracingContext.try_get():
        tracing_context.params_flat = params_flat

    aot_autograd_arg_pos_to_source = None
    # Then, the params 1:1 mapped sources, if relevant.
    if hasattr(gm, "_param_name_to_source"):
        aot_autograd_arg_pos_to_source = []
        # We now know this came from dynamo, and (1) we care about guards,
        # so setting up aot_autograd_arg_pos_to_source for downstream dedup guards
        # can now be done safely. (2) Dynamo logic protects the 1:1 sizing below.
        for name in params.keys():
            assert name in gm._param_name_to_source, f"{name} not found."
            source = gm._param_name_to_source[name]
            assert source not in seen_sources, source
            seen_sources.add(source)
            aot_autograd_arg_pos_to_source.append(source)

    # Next, the input args
    full_args.extend(example_inputs)

    static_input_indices = []
    for pos, node in enumerate(find_nodes(gm.graph, op="placeholder")):
        if hasattr(node, "_dynamo_source"):
            if aot_autograd_arg_pos_to_source is None:
                aot_autograd_arg_pos_to_source = []
            source = node._dynamo_source
            assert source not in seen_sources, source
            seen_sources.add(source)
            aot_autograd_arg_pos_to_source.append(source)
            source_name = source.name() if source else str(source)

            if "tensor_dict" in node.meta and node.meta["tensor_dict"].get(
                "_dynamo_static_input_type", None
            ):
                static_input_indices.append(pos)
            else:
                print("Non-static input pos %s for source %s", pos, source_name)

    if aot_autograd_arg_pos_to_source is not None:
        assert len(full_args) == len(aot_autograd_arg_pos_to_source)

    dynamic_shapes = False
    # Try to infer `dynamic_shapes from inputs and graph nodes
    fake_mode = detect_fake_mode(full_args)
    if (
        fake_mode is None
        and hasattr(gm, "_orig_mod")
        and isinstance(gm._orig_mod, torch.fx.GraphModule)
    ):
        vals = [
            node.meta["val"] for node in gm._orig_mod.graph.nodes if "val" in node.meta
        ]
        fake_mode = detect_fake_mode(vals)
    dynamic_shapes = fake_mode is not None and fake_mode.shape_env is not None

    aot_config = AOTConfig(
        fw_compiler=None,
        bw_compiler=None,
        inference_compiler=None,
        partition_fn=None,
        decompositions={},
        num_params_buffers=len(params_flat),
        aot_id=next(FX_COUNT),
        keep_inference_input_mutations=False,
        dynamic_shapes=dynamic_shapes,
        aot_autograd_arg_pos_to_source=aot_autograd_arg_pos_to_source,
    )
    if hasattr(aot_config, "static_input_indices"):
        # Compatibility for torch version < 2.5
        aot_config.static_input_indices = static_input_indices
    return params_flat, params_spec, full_args, aot_config


def _invoke_dispatcher(
    flat_fn, fake_flat_args, fake_mode, shape_env, aot_config, stage
):
    python_dispatcher_mode = (
        enable_python_dispatcher() if shape_env is not None else nullcontext()
    )

    needs_autograd = stage == FxStage.pregrad
    # See NOTE: [Deferring tensor pack/unpack hooks until runtime]
    # If any saved tensor hooks are active, we **don't** want to trace them.
    # Instead, we'll let them run at runtime, around the custom autograd.Function
    # that we generate in torch.compile.
    with torch.autograd.set_multithreading_enabled(
        False
    ), preserve_rng_state(), fake_mode, python_dispatcher_mode, PhiloxStateTracker():
        with enable_python_dispatcher():
            with patch("torch.cuda.set_rng_state", lambda *args: None):
                if hasattr(aot_config, "static_input_indices"):
                    fw_metadata = run_functionalized_fw_and_collect_metadata(
                        flat_fn,
                        static_input_indices=aot_config.static_input_indices,
                        keep_input_mutations=aot_config.keep_inference_input_mutations,
                        is_train=needs_autograd,
                        pre_dispatch=aot_config.pre_dispatch,
                    )(*fake_flat_args)
                else:
                    fw_metadata = run_functionalized_fw_and_collect_metadata(
                        flat_fn,
                        keep_input_mutations=aot_config.keep_inference_input_mutations,
                        is_train=needs_autograd,
                        pre_dispatch=aot_config.pre_dispatch,
                    )(*fake_flat_args)

        if needs_autograd and not aot_config.pre_dispatch:
            dispatched_fn = aot_dispatch_autograd_graph(
                flat_fn, fake_flat_args, aot_config, fw_metadata=fw_metadata
            )
        else:
            dispatched_fn = aot_dispatch_base_graph(
                flat_fn, fake_flat_args, aot_config, fw_metadata=fw_metadata
            )
        if isinstance(dispatched_fn, tuple):
            dispatched_fn = dispatched_fn[0]
    return dispatched_fn, fw_metadata


def _unlift_params(mod, gm):
    """
    Unlift an exported gm to a STATEFUL graph module.
    Apply mutations on the mutated outputs.
    If freeze, unlift a GraphModule by restoring parameters and buffers from the original module.
    Args:
        mod: The original module containing parameters and buffers
        gm: The GraphModule to be unlifted
        param_fqns: The fully qualified names of parameters to be unlifted

    Returns:
        The unlifted nodes

    The lifted graph inputs is
        [*tokens, *params, *buffers, *user_inputs]
    After unlifting, the graph module will be modified to:
        [*tokens, *user_inputs]
    and the params and buffers would be unlifted to get_attr nodes.
    """
    # Assign parameters to the graph module
    param_fqns = []
    for name, param in mod.named_parameters(remove_duplicate=False):
        _assign_attr(param, gm, name, attr_kind=_AttrKind.PARAMETER)
        param_fqns.append(name)

    # Assign buffers to the graph module
    for name, buffer in mod.named_buffers(remove_duplicate=False):
        _assign_attr(buffer, gm, name, attr_kind=_AttrKind.BUFFER)
        param_fqns.append(name)

    # Unlift inputs as getattr nodes
    unlifted_nodes = []
    placeholder_nodes = [node for node in gm.graph.nodes if node.op == "placeholder"]
    for idx, input_node in enumerate(placeholder_nodes):
        if idx >= len(param_fqns):
            unlifted_nodes.append(input_node)

        else:
            with gm.graph.inserting_after(input_node):
                getattr_node = gm.graph.get_attr(param_fqns[idx])
                input_node.replace_all_uses_with(getattr_node)
                metadata = input_node.meta
                gm.graph.erase_node(input_node)
                getattr_node.meta = metadata
                unlifted_nodes.append(getattr_node)
    return unlifted_nodes


def _insert_mutations(gm, fw_metadata, input_nodes):
    """
    Insert mutations into the graph module.
    The functionalized output is
        [*tokens, *mutations, *user_outputs]
    After inserting mutations, the graph module will be modified to:
        [*tokens, *user_outputs]
    and the mutations would be changed to copy_ nodes.
    """
    # Process input info to identify mutatations
    mutated_inps = []
    for idx, input_info in enumerate(fw_metadata.input_info):
        if input_info.mutates_data:
            mutated_inps.append(input_nodes[idx])

    assert len(mutated_inps) == fw_metadata.num_mutated_inp_runtime_indices

    num_mutated_outs = len(mutated_inps)
    output_node = list(gm.graph.nodes)[-1]
    assert output_node.op == "output"

    outputs = pytree.tree_flatten(output_node.args)[0]

    mutated_outs = outputs[:num_mutated_outs]

    return_nodes_to_copy = {}
    for return_node, inp_node in zip(mutated_outs, mutated_inps):
        with gm.graph.inserting_before(output_node):
            copy_node = gm.graph.call_function(
                torch.ops.aten.copy_.default, (inp_node, return_node)
            )
            return_nodes_to_copy[return_node] = copy_node

    user_output_nodes = outputs[num_mutated_outs:]
    output_args = [
        return_nodes_to_copy[node] if node in return_nodes_to_copy else node
        for node in user_output_nodes
    ]
    with gm.graph.inserting_before(output_node):
        # Only return user outputs
        new_output = gm.graph.output(tuple(output_args))
        new_output.meta.update(output_node.meta)
        output_node.replace_all_uses_with(new_output)
        gm.graph.erase_node(output_node)


def decompose_for_inductor(gm, fake_inputs):
    gm = make_fx(
        gm,
        decomposition_table=torch._inductor.decomposition.select_decomp_table(),
        tracing_mode="fake",
        record_module_stack=True,
    )(*fake_inputs)
    return gm


def has_storage(node: fx.Node) -> bool:
    """We can evaluate only nodes that represent tensors with defined storage."""
    if "val" not in node.meta or not isinstance(node.meta["val"], torch.Tensor):
        return False

    try:
        node.meta["val"].untyped_storage()
    except NotImplementedError:
        return False

    return True


def find_hop_nodes(graph: fx.Graph) -> bool:
    """Check if the graph module contains higher-order operators."""
    hop_nodes = []
    for node in graph.nodes:
        if node.op == "call_function":
            if (
                hasattr(node.target, "namespace")
                and node.target.namespace == "higher_order"
            ):
                hop_nodes.append(node)
    return hop_nodes
