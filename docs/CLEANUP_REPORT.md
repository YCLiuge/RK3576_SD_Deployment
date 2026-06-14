# Cleanup Report

Date: 2026-06-15

## Board Cleanup

The RK3576 board was cleaned after the Dreamshaper LCM pipeline was stabilized.

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

Kept a small, GitHub-friendly project:

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

No file over 100 MB remains in the Git working tree.
