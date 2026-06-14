#!/bin/bash
export PATH=/usr/bin:/home/lzy0x91f/.local/bin:$PATH
cd /home/lzy0x91f/lcm_sd

python3 << 'PYEOF'
import os, time
from rknn.api import RKNN

def compile_onnx(name, onnx_path, out_name, do_quant=False, inputs=None, input_size_list=None):
    print(f"\n{'='*50}")
    print(f"Compiling {name}: {onnx_path}")
    print(f"{'='*50}")
    t0 = time.perf_counter()
    
    rk = RKNN(verbose=False)
    rk.config(target_platform='rk3576')
    
    ret = rk.load_onnx(onnx_path, inputs=inputs, input_size_list=input_size_list)
    if ret != 0:
        print(f"  Load FAILED: {ret}")
        return False
    
    ret = rk.build(do_quantization=do_quant)
    if ret != 0:
        print(f"  Build FAILED: {ret}")
        return False
    
    ret = rk.export_rknn(out_name)
    if ret != 0:
        print(f"  Export FAILED: {ret}")
        return False
    
    sz = os.path.getsize(out_name) / 1024**3
    elapsed = time.perf_counter() - t0
    print(f"  {out_name}: {sz:.2f} GB in {elapsed/60:.1f} min")
    rk.release()
    return True

# 1. Text encoder - static shape (1, 77)
compile_onnx("Text Encoder", "text_encoder/model.onnx", "text_encoder/model.rknn",
             inputs=['input_ids'],
             input_size_list=[[1, 77]])

# 2. UNet - static shapes for 512x512 image (batch=2 for CFG)
compile_onnx("UNet", "unet/model.onnx", "unet/model.rknn",
             inputs=['sample', 'timestep', 'encoder_hidden_states', 'timestep_cond'],
             input_size_list=[[2, 4, 64, 64], [1], [2, 77, 768], [2, 256]])

# 3. VAE decoder - static shape (1, 4, 64, 64) for 512x512
compile_onnx("VAE Decoder", "vae_decoder/model.onnx", "vae_decoder/model.rknn",
             inputs=['latent_sample'],
             input_size_list=[[1, 4, 64, 64]])

print("\nDone compiling all models!")
print("\nResults:")
for f in sorted(os.listdir(".")):
    if f.endswith('.rknn'):
        print(f"  {f}: {os.path.getsize(f)/1024**3:.2f} GB")
PYEOF
