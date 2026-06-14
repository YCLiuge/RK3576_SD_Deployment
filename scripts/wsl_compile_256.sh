#!/bin/bash
export PATH=/usr/bin:/home/lzy0x91f/.local/bin:$PATH
cd /home/lzy0x91f/lcm_sd

python3 << 'PYEOF'
import os, time
from rknn.api import RKNN

def compile_onnx(name, onnx_path, out_name, inputs, input_size_list):
    print(f"\n{'='*50}")
    print(f"Compiling {name} 256x256: {onnx_path}")
    t0 = time.perf_counter()
    
    rk = RKNN(verbose=False)
    rk.config(target_platform='rk3576')
    
    ret = rk.load_onnx(onnx_path, inputs=inputs, input_size_list=input_size_list)
    if ret != 0:
        print(f"  Load FAILED: {ret}")
        return False
    
    ret = rk.build(do_quantization=False)
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

# UNet for 256x256: latent = (2, 4, 32, 32)
compile_onnx("UNet 256", "unet/model.onnx", "unet/model_256.rknn",
    inputs=['sample', 'timestep', 'encoder_hidden_states', 'timestep_cond'],
    input_size_list=[[2, 4, 32, 32], [1], [2, 77, 768], [2, 256]])

# VAE decoder for 256x256: latent = (1, 4, 32, 32)
compile_onnx("VAE 256", "vae_decoder/model.onnx", "vae_decoder/model_256.rknn",
    inputs=['latent_sample'],
    input_size_list=[[1, 4, 32, 32]])

print("\nDone!")
for f in sorted(os.listdir(".")):
    if f.endswith('_256.rknn'):
        print(f"  {f}: {os.path.getsize(f)/1024**3:.2f} GB")
PYEOF
