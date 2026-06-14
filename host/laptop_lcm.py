#!/usr/bin/env python3
"""Run Dreamshaper LCM on laptop (ONNX) as golden reference — same seed/embeds as NPU."""
import numpy as np, os, time
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
ONNX_DIR = os.environ.get("LCM_ONNX_DIR", os.path.join(ROOT, "models", "onnx"))
EMBED_DIR = os.path.join(ROOT, "embeds")
CONFIG_DIR = os.path.join(ROOT, "configs")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

# Load embeddings (same as NPU)
pos_emb = np.load(os.path.join(EMBED_DIR, 'pos_emb.npy'))
neg_emb = np.load(os.path.join(EMBED_DIR, 'neg_emb.npy'))
emb_cat = np.concatenate([neg_emb, pos_emb], axis=0)

# Load ONNX models
import onnxruntime as ort
print('Loading ONNX models...')
unet_sess = ort.InferenceSession(os.path.join(ONNX_DIR, 'unet', 'model.onnx'), providers=['CPUExecutionProvider'])
vae_sess = ort.InferenceSession(os.path.join(ONNX_DIR, 'vae_decoder', 'model.onnx'), providers=['CPUExecutionProvider'])

# Same LCM scheduler
class NumpyLCMScheduler:
    def __init__(self, config=None):
        config = config or {}
        beta_start = config.get('beta_start', 0.00085)
        beta_end = config.get('beta_end', 0.012)
        train_steps = config.get('num_train_timesteps', 1000)
        self.train_steps = train_steps
        betas = np.linspace(beta_start**0.5, beta_end**0.5, train_steps, dtype=np.float64)**2
        self.alphas_cumprod = np.cumprod(1.0 - betas)
        self.original_inference_steps = config.get('original_inference_steps', 50)
        self.timestep_scaling = config.get('timestep_scaling', 10.0)
        self.prediction_type = config.get('prediction_type', 'epsilon')
        
    def set_timesteps(self, num_steps):
        if num_steps > self.train_steps:
            raise ValueError(f"num_steps must be <= {self.train_steps}")
        original_steps = max(self.original_inference_steps, num_steps)
        k = self.train_steps / float(original_steps)
        lcm_origin_timesteps = np.rint(np.asarray(range(1, original_steps + 1)) * k - 1)
        lcm_origin_timesteps = np.clip(lcm_origin_timesteps, 0, self.train_steps - 1).astype(np.int64)
        lcm_origin_timesteps = lcm_origin_timesteps[::-1].copy()
        indices = np.floor(np.linspace(0, len(lcm_origin_timesteps), num=num_steps, endpoint=False)).astype(np.int64)
        self.timesteps = lcm_origin_timesteps[indices].astype(np.int64)
        return self.timesteps
    
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
        if self.prediction_type == 'epsilon':
            pred_x0 = (sample - np.sqrt(beta) * model_output) / np.sqrt(alpha)
        elif self.prediction_type == 'sample':
            pred_x0 = model_output
        elif self.prediction_type == 'v_prediction':
            pred_x0 = np.sqrt(alpha) * sample - np.sqrt(beta) * model_output
        else:
            raise ValueError(f'unsupported prediction_type: {self.prediction_type}')
        c_skip, c_out = self.get_scalings_for_boundary_condition_discrete(t)
        denoised = c_out * pred_x0 + c_skip * sample
        if idx != len(self.timesteps) - 1:
            noise = rng.randn(*model_output.shape).astype(np.float32)
            prev_sample = np.sqrt(alpha_prev) * denoised + np.sqrt(beta_prev) * noise
        else:
            prev_sample = denoised
        return prev_sample.astype(np.float32), denoised.astype(np.float32)

import json
cfg_path = os.path.join(CONFIG_DIR, 'scheduler_config.json')
if os.path.exists(cfg_path):
    with open(cfg_path, encoding='utf-8') as f:
        scheduler = NumpyLCMScheduler(json.load(f))
else:
    scheduler = NumpyLCMScheduler()
NUM_STEPS = 4
CFG = 7.5
timesteps = scheduler.set_timesteps(NUM_STEPS)
print(f'Timesteps: {timesteps}')

# Guidance embedding (same as NPU)
w_raw = np.array([CFG - 1.0], dtype=np.float32) * 1000.0
half = 128
emb_w = np.log(10000.0) / (half - 1)
emb_w = np.exp(np.arange(half, dtype=np.float32) * -emb_w)
emb_w = w_raw[:, None] * emb_w[None, :]
w_emb = np.concatenate([np.sin(emb_w), np.cos(emb_w)], axis=1)
w_emb = np.tile(w_emb, (2, 1)).astype(np.float32)

# Init latent (same seed)
rng = np.random.RandomState(42)
latent = rng.randn(1, 4, 64, 64).astype(np.float32)

print(f'\nLCM ONNX ({NUM_STEPS} steps):')
t0 = time.perf_counter()

for i, t in enumerate(timesteps):
    # CFG batch
    lb = np.concatenate([latent, latent], axis=0)
    
    noise_out = unet_sess.run(None, {
        'sample': lb,
        'timestep': np.array([int(t)], dtype=np.int64),
        'encoder_hidden_states': emb_cat,
        'timestep_cond': w_emb,
    })[0]
    
    eps_neg, eps_pos = noise_out[0:1], noise_out[1:2]
    noise_pred = eps_neg + CFG * (eps_pos - eps_neg)
    
    latent, denoised = scheduler.step(noise_pred, t, latent, rng)
    
    if i == 0 or i == NUM_STEPS - 1:
        print(f'  step {i+1}/{NUM_STEPS}  t={t:4d}  denoised std={denoised.std():.4f}  latent std={latent.std():.4f}')

elapsed = time.perf_counter() - t0
print(f'  Done in {elapsed:.1f}s')

# VAE decode
latent_scaled = latent / 0.18215
img_out = vae_sess.run(None, {'latent_sample': latent_scaled.astype(np.float32)})[0]

img = (img_out[0].transpose(1, 2, 0) + 1.0) / 2.0
img = np.clip(img * 255, 0, 255).astype(np.uint8)
from PIL import Image
os.makedirs(OUTPUT_DIR, exist_ok=True)
Image.fromarray(img).save(os.path.join(OUTPUT_DIR, 'onnx_output.png'))
print(f'Saved onnx_output.png ({img.shape})  range=[{img.min()},{img.max()}]')
