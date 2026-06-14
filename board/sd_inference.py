#!/usr/bin/env python3
"""Legacy-compatible inference entry for agents.

Prefer sd_lcm.py for new calls. This wrapper keeps the old sd_inference.py name.
"""
import sys

from sd_lcm import main


def translate_args(argv):
    args = list(argv)
    if "--tokenize" in args:
        args.remove("--tokenize")
        args.extend(["--tokens", "/tmp/sd_tokens.npz"])
    if "--out" not in args:
        args.extend(["--out", "/tmp/sd_output.png"])
    return args


if __name__ == "__main__":
    main(translate_args(sys.argv[1:]))
