#!/usr/bin/env python3
"""Compatibility wrapper: old 512x512 board entry.

New agents should prefer:
  python3 /home/cat/sd_lcm.py --mode balanced
"""
from sd_lcm import main


if __name__ == "__main__":
    main(["--mode", "balanced", "--cached-embeds", "--out", "/home/cat/npu_output.png"])
