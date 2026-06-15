#!/usr/bin/env python3
"""Unified LCM Stable Diffusion runner for RK3576/RKNPU.

Primary board usage:
  python3 /home/cat/sd_lcm.py --mode fast --prompt "1girl, cat ears"
  python3 /home/cat/sd_lcm.py --mode balanced --cached-embeds
  python3 /home/cat/sd_lcm.py --mode quality --seed 123 --out /home/cat/out.png
"""
import argparse
import json
import os
import time
from datetime import datetime

import numpy as np
from PIL import Image


DEFAULT_MODEL_DIR = os.environ.get("SD_MODEL_DIR", "/home/cat/lcm_sd")
DEFAULT_OUTPUT_DIR = "/home/cat/sd_outputs"
DEFAULT_NEGATIVE = (
    "worst quality, low quality, lowres, bad anatomy, bad hands, text, "
    "watermark, blurry, jpeg artifacts"
)
DEFAULT_PROMPT = (
    "masterpiece, best quality, anime illustration, 1girl, cat ears, "
    "white hair, blue eyes, soft light, detailed face"
)

MODES = {
    "fast": {
        "resolution": 256,
        "steps": 4,
        "cfg": 7.0,
        "description": "Fast 256x256 preview, usually about 6 seconds on RK3576.",
    },
    "balanced": {
        "resolution": 512,
        "steps": 4,
        "cfg": 7.5,
        "description": "Balanced 512x512 generation, usually about 36 seconds on RK3576.",
    },
    "quality": {
        "resolution": 512,
        "steps": 100,
        "cfg": 6.0,
        "description": "Slow 512x512 high-quality generation, intended for final anime images.",
    },
    "quality100": {
        "resolution": 512,
        "steps": 100,
        "cfg": 1.0,
        "description": "Experimental 512x512 high-iteration run, 100 steps.",
    },
    "quality150": {
        "resolution": 512,
        "steps": 150,
        "cfg": 1.0,
        "description": "Experimental 512x512 high-iteration run, 150 steps.",
    },
}


def load_model_info(model_dir):
    path = os.path.join(model_dir, "model_info.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_modes(model_info):
    modes = {name: cfg.copy() for name, cfg in MODES.items()}
    for name, cfg in model_info.get("modes", {}).items():
        if isinstance(cfg, dict):
            merged = modes.get(name, {}).copy()
            merged.update(cfg)
            modes[name] = merged
    return modes


def resolve_unet_input_count(args, model_info):
    if args.unet_inputs != "auto":
        return int(args.unet_inputs)
    if "unet_input_count" in model_info:
        return int(model_info["unet_input_count"])
    names = model_info.get("unet_inputs") or []
    if names:
        return len(names)
    return 4


class NumpyLCMScheduler:
    """Small numpy implementation matching diffusers.LCMScheduler for this pipeline."""

    def __init__(self, config):
        beta_start = config.get("beta_start", 0.00085)
        beta_end = config.get("beta_end", 0.012)
        train_steps = config.get("num_train_timesteps", 1000)
        self.train_steps = train_steps
        betas = np.linspace(beta_start**0.5, beta_end**0.5, train_steps, dtype=np.float64) ** 2
        self.alphas_cumprod = np.cumprod(1.0 - betas)
        self.original_inference_steps = config.get("original_inference_steps", 50)
        self.timestep_scaling = config.get("timestep_scaling", 10.0)
        self.prediction_type = config.get("prediction_type", "epsilon")
        self.timesteps = None

    def set_timesteps(self, num_steps):
        if num_steps < 1:
            raise ValueError("--steps must be positive")
        if num_steps > self.train_steps:
            raise ValueError(f"--steps must be <= {self.train_steps}")
        original_steps = max(self.original_inference_steps, num_steps)
        k = self.train_steps / float(original_steps)
        lcm_origin_timesteps = np.rint(np.asarray(range(1, original_steps + 1)) * k - 1)
        lcm_origin_timesteps = np.clip(lcm_origin_timesteps, 0, self.train_steps - 1).astype(np.int64)
        lcm_origin_timesteps = lcm_origin_timesteps[::-1].copy()
        indices = np.floor(
            np.linspace(0, len(lcm_origin_timesteps), num=num_steps, endpoint=False)
        ).astype(np.int64)
        self.timesteps = lcm_origin_timesteps[indices].astype(np.int64)
        return self.timesteps

    def get_scalings_for_boundary_condition_discrete(self, timestep):
        sigma_data = 0.5
        scaled_timestep = timestep * self.timestep_scaling
        c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
        c_out = scaled_timestep / np.sqrt(scaled_timestep**2 + sigma_data**2)
        return c_skip, c_out

    def step(self, model_output, timestep, sample, rng):
        idx = np.where(self.timesteps == timestep)[0][0]
        if idx < len(self.timesteps) - 1:
            prev_timestep = self.timesteps[idx + 1]
        else:
            prev_timestep = timestep

        alpha = self.alphas_cumprod[timestep]
        alpha_prev = self.alphas_cumprod[prev_timestep]
        beta = 1.0 - alpha
        beta_prev = 1.0 - alpha_prev

        if self.prediction_type == "epsilon":
            pred_x0 = (sample - np.sqrt(beta) * model_output) / np.sqrt(alpha)
        elif self.prediction_type == "sample":
            pred_x0 = model_output
        elif self.prediction_type == "v_prediction":
            pred_x0 = np.sqrt(alpha) * sample - np.sqrt(beta) * model_output
        else:
            raise ValueError(f"Unsupported prediction_type: {self.prediction_type}")

        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(timestep)
        denoised = c_out * pred_x0 + c_skip * sample

        if idx != len(self.timesteps) - 1:
            noise = rng.randn(*model_output.shape).astype(np.float32)
            prev_sample = np.sqrt(alpha_prev) * denoised + np.sqrt(beta_prev) * noise
        else:
            prev_sample = denoised
        return prev_sample.astype(np.float32), denoised.astype(np.float32)


class RKNNModel:
    def __init__(self, path, core_mask="core0"):
        from rknnlite.api import RKNNLite

        self.path = path
        self.rk = RKNNLite()
        ret = self.rk.load_rknn(path)
        if ret != 0:
            raise RuntimeError(f"load_rknn failed for {path}: {ret}")

        if core_mask == "auto":
            ret = self.rk.init_runtime()
        else:
            mask = getattr(RKNNLite, "NPU_CORE_0")
            ret = self.rk.init_runtime(core_mask=mask)
        if ret != 0:
            raise RuntimeError(f"init_runtime failed for {path}: {ret}")

    def infer(self, inputs):
        return self.rk.inference(inputs=inputs)

    def release(self):
        self.rk.release()


def load_scheduler(model_dir):
    path = os.path.join(model_dir, "scheduler", "scheduler_config.json")
    with open(path, encoding="utf-8") as f:
        return NumpyLCMScheduler(json.load(f))


def model_paths(model_dir, resolution):
    if resolution == 256:
        return {
            "latent_res": 32,
            "unet": os.path.join(model_dir, "unet", "model_256.rknn"),
            "vae": os.path.join(model_dir, "vae_decoder", "model_256.rknn"),
        }
    if resolution == 512:
        return {
            "latent_res": 64,
            "unet": os.path.join(model_dir, "unet", "model.rknn"),
            "vae": os.path.join(model_dir, "vae_decoder", "model.rknn"),
        }
    raise ValueError("Only 256 and 512 resolutions are supported by the current RKNN models")


def build_guidance_embedding(cfg):
    w_raw = np.array([cfg - 1.0], dtype=np.float32) * 1000.0
    half = 128
    emb_w = np.log(10000.0) / (half - 1)
    emb_w = np.exp(np.arange(half, dtype=np.float32) * -emb_w)
    emb_w = w_raw[:, None] * emb_w[None, :]
    emb = np.concatenate([np.sin(emb_w), np.cos(emb_w)], axis=1)
    return np.tile(emb, (2, 1)).astype(np.float32)


def build_clip_tokenizer(tokenizer_dir):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers

    tokenizer_json = os.path.join(tokenizer_dir, "tokenizer.json")
    if os.path.exists(tokenizer_json):
        return Tokenizer.from_file(tokenizer_json)

    with open(os.path.join(tokenizer_dir, "vocab.json"), encoding="utf-8") as f:
        vocab = json.load(f)
    with open(os.path.join(tokenizer_dir, "merges.txt"), encoding="utf-8") as f:
        merges = [
            tuple(line.split())
            for line in f.read().splitlines()
            if line.strip() and not line.startswith("#") and len(line.split()) == 2
        ]

    tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="<|endoftext|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    return tokenizer


def encode_clip_ids(tokenizer, text):
    start_id = 49406
    end_id = 49407
    ids = tokenizer.encode(text).ids
    ids = [start_id] + ids[:75] + [end_id]
    ids += [end_id] * (77 - len(ids))
    return np.asarray([ids], dtype=np.int64)


def count_clip_tokens(model_dir, text):
    tokenizer = build_clip_tokenizer(os.path.join(model_dir, "tokenizer"))
    return len(tokenizer.encode(text).ids)


def tokenize_prompts(model_dir, prompt, negative):
    tokenizer = build_clip_tokenizer(os.path.join(model_dir, "tokenizer"))
    return encode_clip_ids(tokenizer, prompt), encode_clip_ids(tokenizer, negative)


def read_embedding_meta(pos_emb_path):
    path = pos_emb_path + ".json"
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_embedding_meta(args, prompt, negative):
    meta = {
        "prompt": prompt,
        "negative": negative,
        "model_dir": args.model_dir,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with open(args.pos_emb + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def load_text_embeddings(args, prompt, negative):
    if args.cached_embeds:
        pos = np.load(args.pos_emb).astype(np.float32)
        neg = np.load(args.neg_emb).astype(np.float32)
        return pos, neg, "cached", read_embedding_meta(args.pos_emb)

    if args.tokens:
        data = np.load(args.tokens)
        pos_ids = data["pos"].astype(np.int64)
        neg_ids = data["neg"].astype(np.int64)
    else:
        pos_ids, neg_ids = tokenize_prompts(args.model_dir, prompt, negative)

    text_encoder_path = os.path.join(args.model_dir, "text_encoder", "model.rknn")
    text_encoder = RKNNModel(text_encoder_path, args.core)
    try:
        pos = text_encoder.infer([pos_ids])[0].astype(np.float32)
        neg = text_encoder.infer([neg_ids])[0].astype(np.float32)
    finally:
        text_encoder.release()

    if args.save_embeds:
        np.save(args.pos_emb, pos)
        np.save(args.neg_emb, neg)
        write_embedding_meta(args, prompt, negative)
    return pos, neg, "text_encoder", {"prompt": prompt, "negative": negative}


def make_output_path(mode):
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return os.path.join(DEFAULT_OUTPUT_DIR, f"sd_{mode}_{stamp}.png")


def decode_image(vae_output):
    if vae_output.ndim != 4:
        raise RuntimeError(f"Unexpected VAE output rank: {vae_output.shape}")
    if vae_output.shape[1] == 3:
        img_arr = vae_output[0].transpose(1, 2, 0)
    elif vae_output.shape[-1] == 3:
        img_arr = vae_output[0]
    else:
        raise RuntimeError(f"Unexpected VAE output shape: {vae_output.shape}")
    return np.clip(((img_arr + 1.0) / 2.0) * 255, 0, 255).astype(np.uint8)


def run(args):
    model_info = load_model_info(args.model_dir)
    modes = get_modes(model_info)
    if args.mode not in modes:
        choices = ", ".join(sorted(modes))
        raise ValueError(f"Unknown --mode {args.mode!r}. Available modes: {choices}")
    mode = modes[args.mode].copy()
    resolution = args.resolution or mode["resolution"]
    steps = args.steps or mode["steps"]
    cfg = args.cfg if args.cfg is not None else mode["cfg"]
    prompt = args.prompt or model_info.get("default_prompt") or DEFAULT_PROMPT
    negative = args.negative or model_info.get("default_negative") or DEFAULT_NEGATIVE
    output = args.out or make_output_path(args.mode)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    unet_input_count = resolve_unet_input_count(args, model_info)
    if unet_input_count not in (3, 4):
        raise ValueError(f"Unsupported UNet input count: {unet_input_count}")

    paths = model_paths(args.model_dir, resolution)
    scheduler = load_scheduler(args.model_dir)
    timesteps = scheduler.set_timesteps(steps)
    rng = np.random.RandomState(args.seed)

    print(f"Mode: {args.mode}  resolution={resolution}  steps={steps}  cfg={cfg}  seed={args.seed}")
    print(f"Timesteps: {timesteps.tolist()}")
    prompt_tokens = count_clip_tokens(args.model_dir, prompt)
    negative_tokens = count_clip_tokens(args.model_dir, negative)
    if prompt_tokens > 75:
        print(f"Warning: prompt is {prompt_tokens} CLIP tokens; only the first 75 content tokens are used.")
    if negative_tokens > 75:
        print(f"Warning: negative prompt is {negative_tokens} CLIP tokens; only the first 75 content tokens are used.")

    t_all = time.perf_counter()
    t0 = time.perf_counter()
    pos_emb, neg_emb, emb_source, embed_meta = load_text_embeddings(args, prompt, negative)
    emb_cat = np.concatenate([neg_emb, pos_emb], axis=0).astype(np.float32)
    text_time = time.perf_counter() - t0
    print(f"Text embeddings: {emb_source}  {text_time:.2f}s")

    w_emb = build_guidance_embedding(cfg) if unet_input_count == 4 else None
    latent_res = paths["latent_res"]
    latent = rng.randn(1, 4, latent_res, latent_res).astype(np.float32)

    print("Loading UNet...")
    unet = RKNNModel(paths["unet"], args.core)
    t0 = time.perf_counter()
    try:
        for i, timestep in enumerate(timesteps):
            latent_batch = np.concatenate([latent, latent], axis=0)
            latent_nhwc = latent_batch.transpose(0, 2, 3, 1).copy()
            unet_inputs = [
                latent_nhwc,
                np.array([int(timestep)], dtype=np.int64),
                emb_cat,
            ]
            if unet_input_count == 4:
                unet_inputs.append(w_emb)
            noise_out = unet.infer(unet_inputs)[0]

            eps_neg, eps_pos = noise_out[0:1], noise_out[1:2]
            if eps_pos.shape == (1, latent_res, latent_res, 4):
                eps_neg = eps_neg.transpose(0, 3, 1, 2)
                eps_pos = eps_pos.transpose(0, 3, 1, 2)
            noise_pred = eps_neg + cfg * (eps_pos - eps_neg)
            latent, denoised = scheduler.step(noise_pred, int(timestep), latent, rng)

            if args.verbose or i == 0 or i == len(timesteps) - 1:
                print(
                    f"  step {i + 1:2d}/{steps}  t={int(timestep):4d}  "
                    f"denoised_std={denoised.std():.4f}  latent_std={latent.std():.4f}"
                )
    finally:
        unet.release()
    unet_time = time.perf_counter() - t0

    print("Loading VAE decoder...")
    vae = RKNNModel(paths["vae"], args.core)
    t0 = time.perf_counter()
    try:
        vae_in = (latent / 0.18215).transpose(0, 2, 3, 1).copy()
        img_out = vae.infer([vae_in])[0]
    finally:
        vae.release()
    vae_time = time.perf_counter() - t0

    image = decode_image(img_out)
    Image.fromarray(image).save(output)
    total_time = time.perf_counter() - t_all

    meta = {
        "mode": args.mode,
        "model_name": model_info.get("name", "unknown"),
        "resolution": resolution,
        "steps": steps,
        "cfg": cfg,
        "seed": args.seed,
        "prompt": prompt,
        "negative": negative,
        "prompt_token_count": prompt_tokens,
        "negative_token_count": negative_tokens,
        "prompt_token_limit": 75,
        "prompt_truncated": prompt_tokens > 75,
        "negative_truncated": negative_tokens > 75,
        "encoded_prompt": embed_meta.get("prompt"),
        "encoded_negative": embed_meta.get("negative"),
        "output": output,
        "timesteps": [int(t) for t in timesteps],
        "unet_input_count": unet_input_count,
        "text_embedding_source": emb_source,
        "text_time_sec": round(text_time, 3),
        "unet_time_sec": round(unet_time, 3),
        "vae_time_sec": round(vae_time, 3),
        "total_time_sec": round(total_time, 3),
        "image_min": int(image.min()),
        "image_max": int(image.max()),
        "image_shape": list(image.shape),
    }
    with open(output + ".json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output}")
    print(f"Times: text={text_time:.1f}s  unet={unet_time:.1f}s  vae={vae_time:.1f}s  total={total_time:.1f}s")
    if args.json:
        print(json.dumps(meta, ensure_ascii=False))
    return meta


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run LCM Stable Diffusion on RK3576 NPU")
    parser.add_argument("--mode", default="balanced")
    parser.add_argument("--list-modes", action="store_true", help="Print mode presets and exit")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--negative", default=None)
    parser.add_argument("--resolution", type=int, choices=[256, 512], default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--cfg", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    parser.add_argument("--cached-embeds", action="store_true", help="Use /home/cat/pos_emb.npy and neg_emb.npy")
    parser.add_argument("--save-embeds", action="store_true", help="Save current prompt embeddings to pos/neg npy")
    parser.add_argument("--pos-emb", default="/home/cat/pos_emb.npy")
    parser.add_argument("--neg-emb", default="/home/cat/neg_emb.npy")
    parser.add_argument("--tokens", default=None, help="Use token npz with pos/neg arrays, e.g. /tmp/sd_tokens.npz")
    parser.add_argument("--unet-inputs", choices=["auto", "3", "4"], default="auto")
    parser.add_argument("--core", choices=["core0", "auto"], default="core0")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print final metadata as one JSON line")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.list_modes:
        for name, cfg in get_modes(load_model_info(args.model_dir)).items():
            print(f"{name}: {cfg}")
        return None
    return run(args)


if __name__ == "__main__":
    main()
