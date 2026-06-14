#!/bin/bash
export PATH=/usr/bin:/home/lzy0x91f/.local/bin:$PATH
python3 << 'PYEOF'
import onnx

m = onnx.load('/home/lzy0x91f/lcm_sd/unet/model.onnx', load_external_data=False)
print('UNet inputs:')
for i in m.graph.input:
    dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
    print(f'  {i.name}: {dims}')
print('UNet outputs:')
for o in m.graph.output:
    dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(f'  {o.name}: {dims}')

m2 = onnx.load('/home/lzy0x91f/lcm_sd/vae_decoder/model.onnx')
print('\nVAE Decoder inputs:')
for i in m2.graph.input:
    dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
    print(f'  {i.name}: {dims}')
print('VAE Decoder outputs:')
for o in m2.graph.output:
    dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(f'  {o.name}: {dims}')

m3 = onnx.load('/home/lzy0x91f/lcm_sd/text_encoder/model.onnx')
print('\nText Encoder inputs:')
for i in m3.graph.input:
    dims = [d.dim_value for d in i.type.tensor_type.shape.dim]
    print(f'  {i.name}: {dims}')
print('Text Encoder outputs:')
for o in m3.graph.output:
    dims = [d.dim_value for d in o.type.tensor_type.shape.dim]
    print(f'  {o.name}: {dims}')
PYEOF
