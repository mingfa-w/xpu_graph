import torch
import torch.nn as nn
import copy
import itertools
import os
from xpu_graph.utils import logger
from xpu_graph.config import get_dump_dir

CASE_CNT = itertools.count()


class AutogradMonitor:
    def __init__(self, golden_mod, mark=None):
        self.golden_mod = golden_mod
        self.mark = mark if mark is not None else "autograd"

        self.saved_states = []
        self.saved_inputs = []
        self.saved_outputs = []

    def _pre_forward_hook(self, module, inputs):
        self.saved_states.append(copy.deepcopy(module.state_dict()))
        forward_inputs = [
            (
                input.detach().requires_grad_(input.requires_grad)
                if isinstance(input, torch.Tensor)
                else input
            )
            for input in inputs
        ]
        self.saved_inputs.append(copy.deepcopy(forward_inputs))

    def _post_forward_hook(self, module, inputs, outputs):
        logger.debug(f"Guarding forward: {self.golden_mod}")
        golden_states = self.saved_states[-1]
        golden_inputs = self.saved_inputs[-1]
        forward_input_golden = {
            "state_dict": copy.deepcopy(golden_states),
            "inputs": copy.deepcopy(golden_inputs),
        }
        self.golden_mod.load_state_dict(golden_states)
        golden_outputs = self.golden_mod.forward(*golden_inputs)
        self.saved_outputs.append(golden_outputs)

        forward_golden = {
            "state_dict": self.golden_mod.state_dict(),
            "inputs": golden_inputs,
            "outputs": golden_outputs,
        }
        forward_target = {
            "state_dict": module.state_dict(),
            "inputs": inputs,
            "outputs": outputs,
        }
        try:
            torch.testing.assert_close(forward_golden, forward_target)
        except AssertionError as e:
            global CASE_CNT
            case_id = next(CASE_CNT)
            dump_path = os.path.join(
                get_dump_dir(), f"case_{self.mark}_forward_{case_id}"
            )
            logger.warning(
                f"The forward pass diverges for {self.golden_mod}\ncases saved_to: {dump_path}\nError: {e}"
            )
            os.makedirs(dump_path, exist_ok=True)
            if isinstance(self.golden_mod, torch.fx.GraphModule):
                with open(os.path.join(dump_path, "golden_mod.py"), "w+t") as gm_f:
                    mod_str = self.golden_mod.print_readable(
                        print_output=False, include_stride=True, include_device=True
                    )
                    gm_f.write(mod_str)

            torch.save(
                forward_input_golden["state_dict"],
                os.path.join(dump_path, "states_golden.pth"),
            )
            torch.save(
                forward_input_golden["inputs"],
                os.path.join(dump_path, "inputs_golden.pth"),
            )

    def _post_backward_hook(self, module, grad_input, grad_output):
        logger.debug(f"Guarding backward: {self.golden_mod}")
        golden_outputs = self.saved_outputs.pop()
        golden_states = self.saved_states.pop()
        golden_inputs = self.saved_inputs.pop()

        backward_input_golden = {
            "state_dict": copy.deepcopy(golden_states),
            "inputs": copy.deepcopy(golden_inputs),
        }

        self.golden_mod.load_state_dict(golden_states)
        for output, grad_output in zip(golden_outputs, grad_output):
            output.backward(grad_output)
        golden_grad_inputs = [
            (
                input.grad
                if input.grad is not None
                else (
                    torch.zeros_like(input) if isinstance(input, torch.Tensor) else None
                )
            )
            for input in golden_inputs
        ]

        backward_golden = {
            "state_dict": golden_states,
            "grad_input": golden_grad_inputs,
        }

        backward_target = {
            "state_dict": module.state_dict(),
            "grad_input": grad_input,
        }

        try:
            torch.testing.assert_close(backward_golden, backward_target)
        except AssertionError as e:
            global CASE_CNT
            case_id = next(CASE_CNT)
            dump_path = os.path.join(
                get_dump_dir(), f"case_{self.mark}_backward_{case_id}"
            )
            logger.warning(
                f"The backward pass diverges for {self.golden_mod}\ncases saved_to: {dump_path}\nError: {e}"
            )
            os.makedirs(dump_path, exist_ok=True)
            if isinstance(self.golden_mod, torch.fx.GraphModule):
                with open(os.path.join(dump_path, "golden_mod.py"), "w+t") as gm_f:
                    mod_str = self.golden_mod.print_readable(
                        print_output=False, include_stride=True, include_device=True
                    )
                    gm_f.write(mod_str)

            torch.save(
                backward_input_golden["state_dict"],
                os.path.join(dump_path, "states_golden.pth"),
            )
            torch.save(
                backward_input_golden["inputs"],
                os.path.join(dump_path, "inputs_golden.pth"),
            )

    def guard(self, mod: nn.Module):
        mod.register_forward_pre_hook(self._pre_forward_hook)
        mod.register_forward_hook(self._post_forward_hook)
        # mod.register_full_backward_pre_hook(self._pre_backward_hook)
        mod.register_full_backward_hook(self._post_backward_hook)


class FunctionMonitor(nn.Module):
    def __init__(self, golden_fn, target_fn, mark=None):
        super(FunctionMonitor, self).__init__()
        self.golden_fn = golden_fn
        self.target_fn = target_fn
        self.mark = mark if mark is not None else "function"
        self.case_cnt = itertools.count()
        golden_fn.print_readable()

    def forward(self, *args):
        assert not torch.is_grad_enabled(), "FunctionMonitor only works in no_grad mode"
        logger.info(f"Guarding function: {self.golden_fn}")
        inputs = copy.deepcopy(
            [
                (input.detach() if isinstance(input, torch.Tensor) else input)
                for input in args
            ]
        )
        golden_inputs = copy.deepcopy(inputs)
        golden_outputs = self.golden_fn(*golden_inputs)
        target_outputs = self.target_fn(*args)
        golden = {
            "inputs": golden_inputs,
            "outputs": golden_outputs,
        }
        target = {
            "inputs": args,
            "outputs": target_outputs,
        }
        try:
            torch.testing.assert_close(golden, target)
        except AssertionError as e:
            case_id = next(self.case_cnt)
            dump_path = os.path.join(get_dump_dir(), f"case_{self.mark}_{case_id}")
            logger.warning(
                f"The function diverges for {self.golden_fn}\ncases saved_to: {dump_path}\nError: {e}"
            )
            os.makedirs(dump_path, exist_ok=True)
            if isinstance(self.golden_fn, torch.fx.GraphModule):
                with open(os.path.join(dump_path, "golden_fn.py"), "w+t") as gm_f:
                    mod_str = self.golden_fn.print_readable(
                        print_output=False, include_stride=True, include_device=True
                    )
                    gm_f.write(mod_str)
            torch.save(
                golden["inputs"],
                os.path.join(dump_path, "inputs_golden.pth"),
            )
        return target_outputs


from torch.utils._python_dispatch import TorchDispatchMode


class OpMonitor(TorchDispatchMode):
    def __init__(self, golden_funcs, mark=None, dispatch_key=None):
        super().__init__(dispatch_key)
        self.golden_funcs = golden_funcs
        self.mark = mark if mark is not None else "op"

    def __torch_dispatch__(self, func, types, args, kwargs=None):
        if func in self.golden_funcs:
            golden_fn = self.golden_funcs[func]
            logger.info(f"Guarding op: {func}")
            inputs = copy.deepcopy(
                [
                    (input.detach() if isinstance(input, torch.Tensor) else input)
                    for input in args
                ]
            )
            golden_inputs = copy.deepcopy(inputs)
            golden_outputs = golden_fn(*golden_inputs, **(kwargs or {}))
            outputs = func(*args, **(kwargs or {}))
            golden = {
                "inputs": golden_inputs,
                "outputs": golden_outputs,
            }
            target = {
                "inputs": args,
                "outputs": outputs,
            }
            try:
                torch.testing.assert_close(golden, target)
            except AssertionError as e:
                global CASE_CNT
                case_id = next(CASE_CNT)
                dump_path = os.path.join(get_dump_dir(), f"case_{self.mark}_{case_id}")
                logger.warning(
                    f"The op diverges for {func}\ncases saved_to: {dump_path}\nError: {e}"
                )
                os.makedirs(dump_path, exist_ok=True)
                torch.save(
                    golden["inputs"],
                    os.path.join(dump_path, "inputs_golden.pth"),
                )
            return outputs
        return func(*args, **(kwargs or {}))
