#!/bin/bash
export PATH=/usr/bin:/home/lzy0x91f/.local/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
WORK=/home/lzy0x91f/lcm_sd
mkdir -p $WORK
cd $WORK

echo "Downloading essential ONNX files..."
python3 << 'PYEOF'
from huggingface_hub import hf_hub_download
import os

repo = "TheyCallMeHex/LCM-Dreamshaper-V7-ONNX"
files = [
    "text_encoder/model.onnx",
    "unet/model.onnx",
    "unet/model.onnx_data",
    "vae_decoder/model.onnx",
    "scheduler/scheduler_config.json",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
]

for f in files:
    local = hf_hub_download(repo, f, local_dir=".", local_dir_use_symlinks=False)
    sz = os.path.getsize(local) / 1024**2
    print(f"  {f} ({sz:.1f} MB)")

# Also download run script
hf_hub_download("thanhtantran/Stable-Diffusion-1.5-LCM-ONNX-RKNN2", "run_rknn-lcm.py", local_dir=".")
print("  run_rknn-lcm.py")
print("\nDONE")
PYEOF
