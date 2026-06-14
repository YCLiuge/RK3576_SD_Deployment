#!/usr/bin/env python3
"""Generate cached Dreamshaper text embeddings with the ONNX text encoder."""
import argparse
import os

import numpy as np
import onnxruntime as ort
from transformers import CLIPTokenizer


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PROMPT = "masterpiece, best quality, 1girl, cat girl, cat ears, white hair, blue eyes, solo, cute"
DEFAULT_NEGATIVE = "lowres, bad anatomy, bad hands, worst quality, low quality"


def main():
    parser = argparse.ArgumentParser(description="Generate pos_emb.npy and neg_emb.npy")
    parser.add_argument("--text-encoder", default=os.path.join(ROOT, "models", "onnx", "text_encoder", "model.onnx"))
    parser.add_argument("--out-dir", default=os.path.join(ROOT, "embeds"))
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative", default=DEFAULT_NEGATIVE)
    parser.add_argument("--tokenizer", default="openai/clip-vit-large-patch14")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer)
    pos_ids = tokenizer(
        args.prompt, padding="max_length", max_length=77, truncation=True, return_tensors="np"
    ).input_ids
    neg_ids = tokenizer(
        args.negative, padding="max_length", max_length=77, truncation=True, return_tensors="np"
    ).input_ids

    sess = ort.InferenceSession(args.text_encoder, providers=["CPUExecutionProvider"])
    pos_emb = sess.run(None, {"input_ids": pos_ids.astype(np.int64)})[0]
    neg_emb = sess.run(None, {"input_ids": neg_ids.astype(np.int64)})[0]

    np.save(os.path.join(args.out_dir, "pos_emb.npy"), pos_emb.astype(np.float32))
    np.save(os.path.join(args.out_dir, "neg_emb.npy"), neg_emb.astype(np.float32))
    print(f"Pos: {pos_emb.shape} std={pos_emb.std():.4f} mean={pos_emb.mean():.4f}")
    print(f"Neg: {neg_emb.shape} std={neg_emb.std():.4f} mean={neg_emb.mean():.4f}")
    print(f"Saved embeddings to {args.out_dir}")


if __name__ == "__main__":
    main()
