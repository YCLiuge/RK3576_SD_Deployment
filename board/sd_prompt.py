#!/usr/bin/env python3
"""Legacy-compatible prompt tokenizer for agents.

This writes /tmp/sd_tokens.npz. The main entry can consume it with:
  python3 /home/cat/sd_lcm.py --tokens /tmp/sd_tokens.npz
"""
import argparse
import os
import numpy as np

from sd_lcm import (
    DEFAULT_MODEL_DIR,
    DEFAULT_NEGATIVE,
    DEFAULT_PROMPT,
    build_clip_tokenizer,
    encode_clip_ids,
)


def main():
    parser = argparse.ArgumentParser(description="Tokenize prompts for sd_lcm.py")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("negative", nargs="?", default=DEFAULT_NEGATIVE)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", default="/tmp/sd_tokens.npz")
    args = parser.parse_args()

    tokenizer = build_clip_tokenizer(os.path.join(args.model_dir, "tokenizer"))
    pos_raw = tokenizer.encode(args.prompt).ids
    neg_raw = tokenizer.encode(args.negative).ids
    pos = encode_clip_ids(tokenizer, args.prompt)
    neg = encode_clip_ids(tokenizer, args.negative)
    np.savez(args.out, pos=pos, neg=neg)
    print(f"Pos: {args.prompt}")
    print(f"Pos tokens: {len(pos_raw)}/75 content tokens")
    print(f"Neg: {args.negative}")
    print(f"Neg tokens: {len(neg_raw)}/75 content tokens")
    if len(pos_raw) > 75:
        print("Warning: positive prompt will be truncated.")
    if len(neg_raw) > 75:
        print("Warning: negative prompt will be truncated.")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
