#!/usr/bin/env python3
"""Run the 256x256 Dreamshaper LCM ONNX pipeline as a golden reference."""
import json
import os
import time

import numpy as np
import onnxruntime as ort
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
ONNX_DIR = os.environ.get("LCM_ONNX_DIR", os.path.join(ROOT, "models", "onnx"))
EMBED_DIR = os.path.join(ROOT, "embeds")
CONFIG_DIR = os.path.join(ROOT, "configs")
OUTPUT_DIR = os.path.join(ROOT, "outputs")
RES = 32
NUM_STEPS = 4
CFG = 7.5


class NumpyLCMScheduler:
    def __init__(self, config):
        beta_start = config.get("beta_start", 0.00085)
        beta_end = config.get("beta_end", 0.012)
        train_steps = config.get("num_train_timesteps", 1000)
        betas = np.linspace(beta_start**0.5, beta_end**0.5, train_steps, dtype=np.float64) ** 2
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas)
        self.original_inference_steps = config.get("original_inference_steps", 50)
        self.timestep_scaling = config.get("timestep_scaling", 10.0)
        self.prediction_type = config.get("prediction_type", "epsilon")
        self.steps_offset = config.get("steps_offset", 1)
        self.timesteps = None

    def set_timesteps(self, num_steps):
        k = len(self.alphas_cumprod) // self.original_inference_steps
        lcm_origin_timesteps = np.asarray(range(1, self.original_inference_steps + 1)) * k - 1
        lcm_origin_timesteps = lcm_origin_timesteps[::-1].copy()
        indices = np.floor(
            np.linspace(0, len(lcm_origin_timesteps), num=num_steps, endpoint=False)
        ).astype(np.int64)
        ts = lcm_origin_timesteps[indices].astype(np.int64)
        self.timesteps = ts
        return ts

    def get_scalings_for_boundary_condition_discrete(self, t):
        sigma_data = 0.5
        scaled_timestep = t * self.timestep_scaling
        c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
        c_out = scaled_timestep / np.sqrt(scaled_timestep**2 + sigma_data**2)
        return c_skip, c_out

    def step(self, model_output, t, sample, rng):
        idx = np.where(self.timesteps == t)[0][0]
        if idx < len(self.timesteps) - 1:
            t_prev = self.timesteps[idx + 1]
            alpha_prev = self.alphas_cumprod[t_prev]
        else:
            t_prev = t
            alpha_prev = self.alphas_cumprod[t_prev]

        alpha = self.alphas_cumprod[t]
        beta = 1.0 - alpha
        beta_prev = 1.0 - alpha_prev

        if self.prediction_type == "epsilon":
            pred_x0 = (sample - np.sqrt(beta) * model_output) / np.sqrt(alpha)
        elif self.prediction_type == "sample":
            pred_x0 = model_output
        elif self.prediction_type == "v_prediction":
            pred_x0 = np.sqrt(alpha) * sample - np.sqrt(beta) * model_output
        else:
            raise ValueError(f"unsupported prediction_type: {self.prediction_type}")

        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(t)
        denoised = c_out * pred_x0 + c_skip * sample

        if idx != len(self.timesteps) - 1:
            noise = rng.randn(*model_output.shape).astype(np.float32)
            prev_sample = np.sqrt(alpha_prev) * denoised + np.sqrt(beta_prev) * noise
        else:
            prev_sample = denoised
        return prev_sample.astype(np.float32), denoised.astype(np.float32)


def load_scheduler():
    path = os.path.join(CONFIG_DIR, "scheduler_config.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return NumpyLCMScheduler(json.load(f))
    return NumpyLCMScheduler({})


def make_guidance_embedding(cfg):
    w_raw = np.array([cfg - 1.0], dtype=np.float32) * 1000.0
    half = 128
    emb_w = np.log(10000.0) / (half - 1)
    emb_w = np.exp(np.arange(half, dtype=np.float32) * -emb_w)
    emb_w = w_raw[:, None] * emb_w[None, :]
    w_emb = np.concatenate([np.sin(emb_w), np.cos(emb_w)], axis=1)
    return np.tile(w_emb, (2, 1)).astype(np.float32)


def main():
    providers = ["CPUExecutionProvider"]
    print("Providers:", ort.get_available_providers())
    print("Loading ONNX models...")
    unet = ort.InferenceSession(os.path.join(ONNX_DIR, "unet", "model.onnx"), providers=providers)
    vae = ort.InferenceSession(os.path.join(ONNX_DIR, "vae_decoder", "model.onnx"), providers=providers)

    pos_emb = np.load(os.path.join(EMBED_DIR, "pos_emb.npy")).astype(np.float32)
    neg_emb = np.load(os.path.join(EMBED_DIR, "neg_emb.npy")).astype(np.float32)
    emb_cat = np.concatenate([neg_emb, pos_emb], axis=0)

    scheduler = load_scheduler()
    timesteps = scheduler.set_timesteps(NUM_STEPS)
    w_emb = make_guidance_embedding(CFG)

    rng = np.random.RandomState(42)
    latent = rng.randn(1, 4, RES, RES).astype(np.float32)

    print(f"\nLCM ONNX 256x256 ({NUM_STEPS} steps, CFG={CFG}):")
    print("Timesteps:", timesteps)
    t0 = time.perf_counter()
    for i, t in enumerate(timesteps):
        lb = np.concatenate([latent, latent], axis=0).astype(np.float32)

        noise_out = unet.run(
            None,
            {
                "sample": lb,
                "timestep": np.array([int(t)], dtype=np.int64),
                "encoder_hidden_states": emb_cat,
                "timestep_cond": w_emb,
            },
        )[0].astype(np.float32)

        eps_neg, eps_pos = noise_out[0:1], noise_out[1:2]
        noise_pred = eps_neg + CFG * (eps_pos - eps_neg)
        latent, denoised = scheduler.step(noise_pred, t, latent, rng)
        if i == 0 or i == NUM_STEPS - 1:
            print(
                f"  step {i + 1}/{NUM_STEPS}  t={t:4d}  "
                f"denoised std={denoised.std():.4f}  latent std={latent.std():.4f}"
            )

    print(f"  Done in {time.perf_counter() - t0:.1f}s")
    img_out = vae.run(None, {"latent_sample": (latent / 0.18215).astype(np.float32)})[0]
    print(f"  VAE output: {img_out.shape}, range=[{img_out.min():.3f},{img_out.max():.3f}]")

    if img_out.shape[1] == 3:
        img_arr = img_out[0].transpose(1, 2, 0)
    else:
        img_arr = img_out[0]
    img = np.clip(((img_arr + 1.0) / 2.0) * 255, 0, 255).astype(np.uint8)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out = os.path.join(OUTPUT_DIR, "onnx_output_256.png")
    Image.fromarray(img).save(out)
    print(f"Saved {out} ({img.shape})  range=[{img.min()},{img.max()}]")


if __name__ == "__main__":
    main()
