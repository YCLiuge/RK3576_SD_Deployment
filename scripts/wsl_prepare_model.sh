#!/bin/bash
set -euo pipefail

SRC=${SRC:-/mnt/d/Electronic_Design/Workspace/Projects/RK_Series/Lubancat3/models/counterfeit_lcm_onnx}
DST=${DST:-/home/lzy0x91f/counterfeit_lcm_sd}

if [ ! -d "$SRC" ]; then
  echo "Missing source directory: $SRC" >&2
  echo "Run host/export_counterfeit_lcm_onnx.py first." >&2
  exit 1
fi

mkdir -p "$DST"
rsync -a --delete "$SRC"/ "$DST"/
echo "Prepared $DST"
