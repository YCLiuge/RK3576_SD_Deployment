# Cleanup Report

Date: 2026-06-15

## Board Cleanup

The RK3576 board was cleaned after the Dreamshaper LCM pipeline was stabilized.
Dreamshaper was later archived to Git LFS and removed from the board. The board
now uses Counterfeit V3.0 + SD1.5 LCM LoRA under `/home/cat/lcm_sd`.

Before cleanup:

```text
/dev/mmcblk1p3  59G total, 44G used, 13G available, 78% used
/home/cat       about 29G
```

Removed project/debug artifacts:

```text
/home/cat/check_external_data     about 3.3G
/home/cat/sd_models               about 3.7G, old Counterfeit/TAESD pipeline
/home/cat/onnx_ref
/home/cat/pytorch_ref
/home/cat/tmp/pip-*
/home/cat/.cache/huggingface      about 4.9G
/home/cat/.cache/modelscope       about 1.9G
/home/cat/.cache/pip              about 2.2G
```

Removed old experimental board scripts:

```text
download_unet.py
compile_sd_models.py
compile_unet*.py
compile_rknn.py
compile_w4a16.py
export_unet*.py
sd_tokenize.py
board_verify.py
board_cmp.py
board_sd.py
board_lcm_test.py
board_vae_test.py
```

Kept runtime-critical files:

```text
/home/cat/lcm_sd/
/home/cat/sd_lcm.py
/home/cat/sd_prompt.py
/home/cat/sd_inference.py
/home/cat/board_lcm.py
/home/cat/board_lcm_256.py
/home/cat/board_gen_emb.py
/home/cat/board_diag.py
/home/cat/pos_emb.npy
/home/cat/neg_emb.npy
/home/cat/pos_emb.npy.json
/home/cat/sd_outputs/
```

After cleanup:

```text
/dev/mmcblk1p3  59G total, 28G used, 29G available, 49% used
/home/cat       about 13G
```

## Local Cleanup

Removed local large files and stale experiment outputs, including:

```text
*.onnx
*.onnx_data
*.rknn
unet/
vae_decoder/
onnx_ref/
pytorch_ref/
venv_rknn/
generated output PNG files
old Counterfeit/DDIM scripts
```

Kept a GitHub-friendly project:

```text
board/
host/
scripts/
configs/
embeds/
assets/
README.md
AGENT.md
requirements-*.txt
```

Compiled Dreamshaper RKNN models were later backed up under:

```text
models/dreamshaper_lcm_rknn/
```

Those files are tracked with Git LFS, not as normal Git blobs.

## Counterfeit Replacement

After the Dreamshaper backup was confirmed on GitHub, the old board model
directory was removed:

```text
/home/cat/lcm_sd  about 4.3G, old Dreamshaper deployment
```

Then the new Counterfeit deployment was uploaded to the same runtime path:

```text
/home/cat/lcm_sd/model_info.json
/home/cat/lcm_sd/text_encoder/model.rknn
/home/cat/lcm_sd/unet/model.rknn
/home/cat/lcm_sd/unet/model_256.rknn
/home/cat/lcm_sd/vae_decoder/model.rknn
/home/cat/lcm_sd/vae_decoder/model_256.rknn
/home/cat/lcm_sd/scheduler/scheduler_config.json
/home/cat/lcm_sd/tokenizer/
```

Current verified disk state after the Counterfeit upload:

```text
/dev/mmcblk1p3  59G total, 28G used, 29G available, 49% used
/home/cat/lcm_sd  about 4.3G
```

Verified board runs:

```text
fast      256x256, 4 steps,  CFG 1.0, about 24s to 45s
balanced  512x512, 8 steps,  CFG 1.0, about 154s
quality   512x512, 12 steps, CFG 1.2, about 204s
```
