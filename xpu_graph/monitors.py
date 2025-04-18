import torch
import torch.nn as nn
import copy
import itertools
import os
from xpu_graph.utils import logger
from xpu_graph.config import get_dump_dir


class AutogradMonitor:
    def __init__(self, golden_mod, mark=0):
        self.golden_mod = golden_mod
        self.mark = mark
        self.case_cnt = itertools.count()

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
            case_id = next(self.case_cnt)
            dump_path = os.path.join(get_dump_dir(), f"case_forward_{case_id}")
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
            case_id = next(self.case_cnt)
            dump_path = os.path.join(get_dump_dir(), f"case_backward_{case_id}")
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
