#!/bin/bash
export PATH=/usr/bin:/home/lzy0x91f/.local/bin:$PATH
export HF_ENDPOINT=https://hf-mirror.com
python3 << 'PYEOF'
from huggingface_hub import list_repo_files
files = list_repo_files("TheyCallMeHex/LCM-Dreamshaper-V7-ONNX")
for f in sorted(files):
    print(f)
print(f"\n{len(files)} files total")
PYEOF
