import torch
import xpu_graph
from xpu_graph.compiler import XpuGraph
from xpu_graph.config import XpuGraphConfig
from xpu_graph.config import OptLevel
import math
import torch.fx as fx
import time
import argparse
import torch
from torch_mlu.utils.model_transfer import transfer
from diffusers import StableDiffusion3Pipeline


def get_model(model_name, dtype, device):
    pipe = StableDiffusion3Pipeline.from_pretrained(model_name)
    pipe.to(device).to(dtype)
    return pipe


def pred(pipe, prompt, output_path):
    """
    config = XpuGraphConfig()
    config.target = xpu_graph.config.Target.mlu
    config.opt_level = OptLevel.level2
    #config.vendor_compiler={"mode":"reduce-overhead"}
    #xpu_graph.debug=True
    xpu_graph = XpuGraph(config)
    pipe = torch.compile(pipe, backend=xpu_graph, dynamic=False)
    """

    image = pipe(
        prompt=prompt,
        negative_prompt="",
        num_inference_steps=28,
        height=1024,
        width=1024,
        guidance_scale=7.0,
    ).images[0]

    image.save(output_path)


def get_args():
    parser = argparse.ArgumentParser(description="Run a SD model on a specified device")

    parser.add_argument(
        "--device",
        type=str,
        required=True,
        default="cpu",
        help="Device to run the model on.",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=False,
        default="./stable-diffusion-3-medium-diffusers/",
        help="Path to the model file.",
    )

    parser.add_argument(
        "--model_dtype",
        type=torch.dtype,
        required=False,
        default=torch.float16,
        help="Model precision.",
    )

    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default="./sd3_hello_world.png",
        help="Path to the model file.",
    )
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    model = get_model(args.model_path, args.model_dtype, args.device)
    prompt = "a photo of a cat holding a sign that says hello world"
    pred(model, prompt, args.output_path)
