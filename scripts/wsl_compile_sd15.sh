#!/bin/bash
set -euo pipefail

export PATH=/usr/bin:/home/lzy0x91f/.local/bin:$PATH
MODEL_DIR=${MODEL_DIR:-/home/lzy0x91f/counterfeit_lcm_sd}
cd "$MODEL_DIR"

python3 << 'PYEOF'
import json
import os
import time

import onnx
from rknn.api import RKNN


def read_graph_inputs(path):
    model = onnx.load(path, load_external_data=False)
    initializers = {i.name for i in model.graph.initializer}
    return [i.name for i in model.graph.input if i.name not in initializers]


def compile_onnx(name, onnx_path, out_name, inputs, input_size_list, do_quant=False):
    print(f"\n{'=' * 60}")
    print(f"Compiling {name}: {onnx_path}")
    print(f"Inputs: {list(zip(inputs, input_size_list))}")
    print(f"{'=' * 60}")
    t0 = time.perf_counter()

    rk = RKNN(verbose=False)
    rk.config(target_platform="rk3576")

    ret = rk.load_onnx(onnx_path, inputs=inputs, input_size_list=input_size_list)
    if ret != 0:
        raise RuntimeError(f"load_onnx failed for {name}: {ret}")

    ret = rk.build(do_quantization=do_quant)
    if ret != 0:
        raise RuntimeError(f"build failed for {name}: {ret}")

    ret = rk.export_rknn(out_name)
    if ret != 0:
        raise RuntimeError(f"export_rknn failed for {name}: {ret}")

    elapsed = time.perf_counter() - t0
    size = os.path.getsize(out_name) / 1024**3
    print(f"{out_name}: {size:.2f} GB in {elapsed / 60:.1f} min")
    rk.release()


def unet_shapes(input_names, latent_hw):
    shapes = {
        "sample": [2, 4, latent_hw, latent_hw],
        "timestep": [1],
        "encoder_hidden_states": [2, 77, 768],
        "timestep_cond": [2, 256],
    }
    return [shapes[name] for name in input_names]


model_info = {}
if os.path.exists("model_info.json"):
    with open("model_info.json", encoding="utf-8") as f:
        model_info = json.load(f)

unet_inputs = model_info.get("unet_inputs") or read_graph_inputs("unet/model.onnx")
vae_inputs = read_graph_inputs("vae_decoder/model.onnx")
text_inputs = read_graph_inputs("text_encoder/model.onnx")

compile_onnx(
    "Text Encoder",
    "text_encoder/model.onnx",
    "text_encoder/model.rknn",
    inputs=text_inputs,
    input_size_list=[[1, 77]],
)

compile_onnx(
    "UNet 512",
    "unet/model.onnx",
    "unet/model.rknn",
    inputs=unet_inputs,
    input_size_list=unet_shapes(unet_inputs, 64),
)

compile_onnx(
    "VAE Decoder 512",
    "vae_decoder/model.onnx",
    "vae_decoder/model.rknn",
    inputs=vae_inputs,
    input_size_list=[[1, 4, 64, 64]],
)

compile_onnx(
    "UNet 256",
    "unet/model.onnx",
    "unet/model_256.rknn",
    inputs=unet_inputs,
    input_size_list=unet_shapes(unet_inputs, 32),
)

compile_onnx(
    "VAE Decoder 256",
    "vae_decoder/model.onnx",
    "vae_decoder/model_256.rknn",
    inputs=vae_inputs,
    input_size_list=[[1, 4, 32, 32]],
)

print("\nDone compiling RKNN models.")
for root, _, files in os.walk("."):
    for name in sorted(files):
        if name.endswith(".rknn"):
            path = os.path.join(root, name)
            print(f"  {path}: {os.path.getsize(path) / 1024**3:.2f} GB")
PYEOF
