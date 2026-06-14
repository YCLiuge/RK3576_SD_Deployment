#!/usr/bin/env python3
"""Compatibility wrapper: old 256x256 board entry.

New agents should prefer:
  python3 /home/cat/sd_lcm.py --mode fast
"""
from sd_lcm import main


if __name__ == "__main__":
    main(["--mode", "fast", "--cached-embeds", "--out", "/home/cat/npu_output.png"])
