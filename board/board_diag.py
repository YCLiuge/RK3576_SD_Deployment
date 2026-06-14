#!/usr/bin/env python3
"""Local board diagnostic for the RK3576 Stable Diffusion deployment."""
import os
import subprocess


def run(title, cmd):
    print(f"\n=== {title} ===")
    result = subprocess.run(cmd, shell=True, text=True, capture_output=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        print("ERR:", result.stderr.strip())
    return result.returncode


def main():
    run("system", "uname -a; df -h /home/cat; free -h | head -2")
    run(
        "python packages",
        "python3 - <<'PY'\n"
        "import numpy\n"
        "print('numpy', numpy.__version__)\n"
        "try:\n"
        "    from PIL import Image\n"
        "    print('Pillow OK')\n"
        "except Exception as exc:\n"
        "    print('Pillow FAIL', exc)\n"
        "try:\n"
        "    from tokenizers import Tokenizer\n"
        "    print('tokenizers OK')\n"
        "except Exception as exc:\n"
        "    print('tokenizers FAIL', exc)\n"
        "try:\n"
        "    from rknnlite.api import RKNNLite\n"
        "    print('rknnlite OK')\n"
        "except Exception as exc:\n"
        "    print('rknnlite FAIL', exc)\n"
        "PY",
    )
    run(
        "rknn runtime",
        "ls -lh /usr/lib/librknnrt.so /usr/bin/rknn_server 2>/dev/null || true; "
        "strings /usr/lib/librknnrt.so 2>/dev/null | grep -E 'librknnrt version|^[0-9]+\\.[0-9]+\\.[0-9]+' | head -5 || true; "
        "strings /usr/bin/rknn_server 2>/dev/null | grep -E 'rknn_server version|^[0-9]+\\.[0-9]+\\.[0-9]+' | head -5 || true",
    )
    run(
        "deployment files",
        "find /home/cat/lcm_sd -maxdepth 3 -type f "
        "\\( -name '*.rknn' -o -name '*.json' -o -name 'merges.txt' -o -name 'vocab.json' \\) "
        "-printf '%p  %s bytes\\n' | sort; "
        "ls -lh /home/cat/sd_lcm.py /home/cat/sd_prompt.py /home/cat/sd_inference.py "
        "/home/cat/board_lcm.py /home/cat/board_lcm_256.py 2>/dev/null",
    )
    run(
        "model load smoke test",
        "python3 - <<'PY'\n"
        "from rknnlite.api import RKNNLite\n"
        "paths = [\n"
        "    '/home/cat/lcm_sd/text_encoder/model.rknn',\n"
        "    '/home/cat/lcm_sd/unet/model_256.rknn',\n"
        "    '/home/cat/lcm_sd/vae_decoder/model_256.rknn',\n"
        "    '/home/cat/lcm_sd/unet/model.rknn',\n"
        "    '/home/cat/lcm_sd/vae_decoder/model.rknn',\n"
        "]\n"
        "for path in paths:\n"
        "    rk = RKNNLite()\n"
        "    ret = rk.load_rknn(path)\n"
        "    print(path, 'load', ret)\n"
        "    if ret == 0:\n"
        "        ret = rk.init_runtime(core_mask=RKNNLite.NPU_CORE_0)\n"
        "        print(path, 'init', ret)\n"
        "    rk.release()\n"
        "PY",
    )


if __name__ == "__main__":
    main()
