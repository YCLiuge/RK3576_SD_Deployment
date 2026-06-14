#!/usr/bin/env python3
"""Legacy-compatible prompt tokenizer for agents.

This writes /tmp/sd_tokens.npz. The main entry can consume it with:
  python3 /home/cat/sd_lcm.py --tokens /tmp/sd_tokens.npz
"""
import argparse
import numpy as np

from sd_lcm import DEFAULT_MODEL_DIR, DEFAULT_NEGATIVE, DEFAULT_PROMPT, tokenize_prompts


def main():
    parser = argparse.ArgumentParser(description="Tokenize prompts for sd_lcm.py")
    parser.add_argument("prompt", nargs="?", default=DEFAULT_PROMPT)
    parser.add_argument("negative", nargs="?", default=DEFAULT_NEGATIVE)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", default="/tmp/sd_tokens.npz")
    args = parser.parse_args()

    pos, neg = tokenize_prompts(args.model_dir, args.prompt, args.negative)
    np.savez(args.out, pos=pos, neg=neg)
    print(f"Pos: {args.prompt}")
    print(f"Neg: {args.negative}")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
