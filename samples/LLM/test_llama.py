from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch_mlu
import xpu_graph
from xpu_graph.compiler import XpuGraph
from xpu_graph.config import XpuGraphConfig
from xpu_graph.config import OptLevel
import math
import torch.fx as fx
import time
import argparse
import torch


def infer_loop(model, input, warmup=50, loop=1000):
    for i in range(warmup):
        result_xpu_graph = model(input)
    torch.mlu.synchronize()

    t1 = time.time()
    for i in range(loop):
        res1 = model(input)
    torch.mlu.synchronize()
    t2 = time.time()
    return (t2 - t1) * 1000.0 / loop  # ms


def get_model(model_name, dtype, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def pred(model, tokenizer, input_text):
    model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        config = xpu_graph.config.XpuGraphConfig()
        config.target = xpu_graph.config.Target.mlu
        config.opt_level = OptLevel.level2
        # config.vendor_compiler = {"mode": "reduce-overhead"}
        xpu_graph_ = xpu_graph.compiler.XpuGraph(config)
        compiled = torch.compile(model, backend=xpu_graph_, dynamic=True)

        result_eager = model(model_inputs.input_ids)
        result_xpu_graph = compiled(model_inputs.input_ids)
        print("result_eager:", result_eager.logits)
        print("result_xpu_graph:", result_xpu_graph.logits)

        dur = infer_loop(model, model_inputs.input_ids)
        print(f"Eager cost {dur} ms")

        dur = infer_loop(compiled, model_inputs.input_ids)
        print(f"Xpu_graph cost {dur} ms")


def get_args():
    parser = argparse.ArgumentParser(
        description="Run a transformers model on a specified device"
    )

    parser.add_argument(
        "--device", type=str, default="cpu", help="Device to run the model on."
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        default="./Llama-2-7b-hf/",
        help="Path to the model file.",
    )

    parser.add_argument(
        "--model_dtype",
        type=str,
        required=False,
        default="float16",
        help="Model precision.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    model, tokenizer = get_model(args.model_path, args.model_dtype, args.device)
    input_text = "Born in north-east France, Soyer trained as a"
    pred(model, tokenizer, input_text)
