#!/bin/bash
set -euo pipefail

export PATH=/usr/bin:/home/lzy0x91f/.local/bin:$PATH

MODEL_DIR=${MODEL_DIR:-/home/lzy0x91f/counterfeit_lcm_sd}
REMOTE_MODEL_DIR=${REMOTE_MODEL_DIR:-/home/cat/lcm_sd}
HOST=${HOST:-10.138.103.190}
BOARD_USER=${BOARD_USER:-cat}
PASS=${PASS:-2335}
export MODEL_DIR REMOTE_MODEL_DIR HOST BOARD_USER PASS

pip install --user paramiko -q 2>/dev/null || true

python3 << 'PYEOF'
import os
import posixpath

import paramiko

local_dir = os.environ["MODEL_DIR"]
remote_dir = os.environ["REMOTE_MODEL_DIR"]
host = os.environ["HOST"]
user = os.environ["BOARD_USER"]
password = os.environ["PASS"]

required = [
    "text_encoder/model.rknn",
    "unet/model.rknn",
    "unet/model_256.rknn",
    "vae_decoder/model.rknn",
    "vae_decoder/model_256.rknn",
    "scheduler/scheduler_config.json",
    "model_info.json",
]
tokenizer_files = [
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
    "tokenizer/tokenizer.json",
]
files = required + [f for f in tokenizer_files if os.path.exists(os.path.join(local_dir, f))]

missing = [f for f in required if not os.path.exists(os.path.join(local_dir, f))]
if missing:
    raise SystemExit(f"Missing required files under {local_dir}: {missing}")

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, username=user, password=password, timeout=10)
sftp = ssh.open_sftp()

def mkdir_p(path):
    parts = path.strip("/").split("/")
    current = ""
    for part in parts:
        current += "/" + part
        try:
            sftp.mkdir(current)
        except OSError:
            pass

try:
    mkdir_p(remote_dir)
    for sub in ["text_encoder", "unet", "vae_decoder", "scheduler", "tokenizer"]:
        mkdir_p(posixpath.join(remote_dir, sub))

    for rel in files:
        local = os.path.join(local_dir, rel)
        remote = posixpath.join(remote_dir, rel.replace("\\", "/"))
        size = os.path.getsize(local) / 1024**2
        print(f"Uploading {rel} ({size:.1f} MB)")
        sftp.put(local, remote)
finally:
    sftp.close()
    ssh.close()

print(f"Uploaded model to {remote_dir}")
PYEOF
