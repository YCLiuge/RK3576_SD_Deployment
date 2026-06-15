# RK3576 Stable Diffusion LCM 部署教程

本项目记录在 **Lubancat3 / RK3576** 上用 **RKNPU** 部署 Stable Diffusion 1.5 系二次元模型的完整流程。当前板端已从旧的 Dreamshaper V7 LCM 切换为：

```text
Counterfeit V3.0 + latent-consistency/lcm-lora-sdv1-5
```

旧的 Dreamshaper RKNN 模型已经从板子删除，节省空间；编译好的旧模型通过 Git LFS 保留在 `models/dreamshaper_lcm_rknn/`，需要时可以恢复。

## 当前状态

| 项目 | 状态 |
|---|---|
| 板子 | Lubancat3 / RK3576 / 8GB RAM |
| 当前板端模型 | Counterfeit V3.0 + SD1.5 LCM LoRA |
| 旧模型 | Dreamshaper V7 LCM 已从板子删除，Git LFS 归档 |
| 运行库 | rknn-toolkit-lite2 2.3.2, librknnrt 2.3.2 |
| 统一入口 | `/home/cat/sd_lcm.py` |
| 模型目录 | `/home/cat/lcm_sd` |
| 输出目录 | `/home/cat/sd_outputs` |
| 板端空间 | 根分区约 59G，总体约 49% 使用率 |

已实测：

| 模式 | 分辨率 | 步数 | CFG | 板端耗时 | 用途 |
|---|---:|---:|---:|---:|---|
| fast | 256x256 | 4 | 1.0 | 约 24 到 45 秒 | 快速预览 |
| balanced | 512x512 | 8 | 1.0 | 约 154 秒 | 默认推荐 |
| quality | 512x512 | 12 | 1.2 | 约 204 秒 | 更慢的最终图 |

LCM LoRA 和普通 SD 不一样，CFG 不宜太高。Counterfeit 这套默认用 `1.0` 到 `1.2`，如果画面发灰或不听 prompt，可以先试 `--cfg 1.5`，不要直接拉到 7 或 8。

## 一句话运行

在板子上：

```bash
python3 /home/cat/sd_lcm.py --mode balanced --prompt "masterpiece, best quality, anime illustration, 1girl, white hair, blue eyes, detailed eyes, soft light" --json
```

输出图片和同名 JSON 元数据会保存在：

```text
/home/cat/sd_outputs/
```

## 三种模式

```bash
# 快速看构图
python3 /home/cat/sd_lcm.py --mode fast --prompt "masterpiece, best quality, anime girl"

# 默认推荐
python3 /home/cat/sd_lcm.py --mode balanced --prompt "masterpiece, best quality, anime illustration, 1girl, blue eyes"

# 更慢的最终图
python3 /home/cat/sd_lcm.py --mode quality --seed 42 --out /home/cat/sd_outputs/final.png \
  --prompt "masterpiece, best quality, anime illustration, 1girl, white hair, blue eyes, detailed eyes, soft light"
```

可以覆盖预设：

```bash
python3 /home/cat/sd_lcm.py --mode balanced --steps 10 --cfg 1.2 --seed 123 --prompt "..."
```

## Prompt 长度

这是 SD1.5 / CLIP tokenizer，最大长度是 77 token。扣掉开始和结束 token，实际 prompt 内容大约 75 token。超过的部分会被截断。

建议写短而清楚的二次元 prompt：

```text
masterpiece, best quality, anime illustration, 1girl, white hair, blue eyes, detailed eyes, soft light
```

负向 prompt 推荐：

```text
worst quality, low quality, lowres, bad anatomy, bad hands, text, watermark, blurry
```

## 项目结构

```text
.
├── board/                         # 上传到板子 /home/cat 后直接运行
│   ├── sd_lcm.py                  # 统一入口，推荐 agent 调用
│   ├── sd_prompt.py               # 只做 tokenize，输出 /tmp/sd_tokens.npz
│   ├── sd_inference.py            # 兼容旧接口
│   ├── board_lcm.py               # 兼容旧 512 入口
│   ├── board_lcm_256.py           # 兼容旧 256 入口
│   ├── board_gen_emb.py           # 生成 pos_emb.npy / neg_emb.npy
│   └── board_diag.py              # 板端诊断
├── host/                          # Windows 电脑端脚本
│   ├── deploy_board.py            # 上传 board/ 脚本到板子
│   ├── export_counterfeit_lcm_onnx.py
│   ├── remote_board_diag.py
│   └── laptop_lcm*.py             # 旧 ONNX golden reference
├── scripts/                       # WSL 下载、复制、编译、上传 RKNN
│   ├── wsl_prepare_model.sh
│   ├── wsl_compile_sd15.sh
│   └── wsl_upload_model.sh
├── configs/
│   ├── scheduler_config.json      # 旧参考配置
│   └── sd15_v1_inference.yaml     # SD1.5 单文件 checkpoint 加载配置
├── models/
│   └── dreamshaper_lcm_rknn/      # 旧 Dreamshaper RKNN 的 Git LFS 归档
├── docs/
└── AGENT.md                       # 给板端 agent 的操作说明
```

生成的 Counterfeit checkpoint、ONNX、RKNN 中间产物默认不进 Git。

## 板端文件布局

当前板子上核心文件：

```text
/home/cat/
├── sd_lcm.py
├── sd_prompt.py
├── sd_inference.py
├── board_lcm.py
├── board_lcm_256.py
├── board_gen_emb.py
├── board_diag.py
├── pos_emb.npy                    # 可选，当前模型的缓存 embedding
├── pos_emb.npy.json               # embedding 对应的 prompt 记录
├── neg_emb.npy
├── sd_outputs/
└── lcm_sd/
    ├── model_info.json
    ├── text_encoder/model.rknn
    ├── unet/model_256.rknn
    ├── unet/model.rknn
    ├── vae_decoder/model_256.rknn
    ├── vae_decoder/model.rknn
    ├── scheduler/scheduler_config.json
    └── tokenizer/{merges.txt,vocab.json,...}
```

`model_info.json` 会告诉 `sd_lcm.py` 当前模型是 3 输入 UNet 还是 4 输入 UNet，并覆盖 fast/balanced/quality 默认参数。

## 常用参数

```bash
python3 /home/cat/sd_lcm.py --help
```

| 参数 | 说明 |
|---|---|
| `--mode fast|balanced|quality` | 选择预设模式 |
| `--prompt "..."` | 正向提示词 |
| `--negative "..."` | 负向提示词 |
| `--seed 42` | 随机种子，同配置下可复现 |
| `--cfg 1.0` | 提示词引导强度，LCM LoRA 通常 1.0 到 1.5 |
| `--steps 8` | 推理步数，LCM 通常 4 到 12 步 |
| `--resolution 256|512` | 覆盖模式分辨率 |
| `--out path.png` | 指定输出图片 |
| `--cached-embeds` | 使用 `/home/cat/pos_emb.npy` 和 `neg_emb.npy` |
| `--save-embeds` | 将当前 prompt 的 embedding 保存为缓存 |
| `--tokens /tmp/sd_tokens.npz` | 使用 `sd_prompt.py` 预先 tokenize 的结果 |
| `--unet-inputs auto|3|4` | 手动指定 UNet 输入数量，默认自动 |
| `--json` | 输出一行 JSON，方便 agent 解析 |

## Agent 调用方式

最推荐：

```bash
python3 /home/cat/sd_lcm.py --mode balanced --prompt "masterpiece, best quality, anime girl" --json
```

如果要分两步控制 prompt：

```bash
python3 /home/cat/sd_prompt.py "masterpiece, best quality, anime girl" "low quality, bad anatomy"
python3 /home/cat/sd_lcm.py --mode fast --tokens /tmp/sd_tokens.npz --json
```

如果要复用当前缓存 embedding：

```bash
python3 /home/cat/sd_lcm.py --mode fast --cached-embeds --json
```

注意：缓存 embedding 只代表保存时的 prompt。现在会额外写 `/home/cat/pos_emb.npy.json`，agent 可以读取它确认缓存内容。

## 从零部署 Counterfeit V3.0 + LCM LoRA

以下命令默认在本项目根目录运行：

```powershell
cd D:\Electronic_Design\Workspace\Projects\RK_Series\Lubancat3
```

### 1. 准备 host 环境

使用用户指定的 conda 环境：

```powershell
conda activate comprehensive
D:\Anaconda3\envs\comprehensive\python.exe -m pip install -r requirements-host.txt
```

如果遇到 OpenMP 重复初始化：

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
```

### 2. 下载模型权重

推荐文件：

```text
gsdf/Counterfeit-V3.0/Counterfeit-V3.0_fix_fp16.safetensors
latent-consistency/lcm-lora-sdv1-5/pytorch_lora_weights.safetensors
```

PowerShell 断点续传示例：

```powershell
New-Item -ItemType Directory -Force models\checkpoints\counterfeit_v30
New-Item -ItemType Directory -Force models\checkpoints\lcm_lora_sdv15

curl.exe -L -C - -o models\checkpoints\counterfeit_v30\Counterfeit-V3.0_fix_fp16.safetensors `
  https://hf-mirror.com/gsdf/Counterfeit-V3.0/resolve/main/Counterfeit-V3.0_fix_fp16.safetensors

curl.exe -L -C - -o models\checkpoints\lcm_lora_sdv15\pytorch_lora_weights.safetensors `
  https://hf-mirror.com/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors
```

本次实测官方 Hugging Face 直连不稳定，`hf-mirror.com` 更顺。

### 3. 导出 ONNX

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"
D:\Anaconda3\envs\comprehensive\python.exe host\export_counterfeit_lcm_onnx.py --device cuda --dtype fp16
```

输出：

```text
models/counterfeit_lcm_onnx/
├── model_info.json
├── text_encoder/model.onnx
├── unet/model.onnx
├── vae_decoder/model.onnx
├── scheduler/scheduler_config.json
└── tokenizer/
```

这版 UNet 是 3 输入：

```text
sample, timestep, encoder_hidden_states
```

旧 Dreamshaper LCM 蒸馏版是 4 输入，多了 `timestep_cond`。`sd_lcm.py` 已兼容两种形式。

### 4. 复制到 WSL

```powershell
wsl.exe bash -lc "cd /mnt/d/Electronic_Design/Workspace/Projects/RK_Series/Lubancat3 && bash scripts/wsl_prepare_model.sh"
```

默认复制到：

```text
/home/lzy0x91f/counterfeit_lcm_sd
```

### 5. 编译 RKNN

```powershell
wsl.exe bash -lc "cd /mnt/d/Electronic_Design/Workspace/Projects/RK_Series/Lubancat3 && MODEL_DIR=/home/lzy0x91f/counterfeit_lcm_sd bash scripts/wsl_compile_sd15.sh"
```

本次实测编译结果：

```text
text_encoder/model.rknn       0.24 GB
unet/model.rknn               1.88 GB
unet/model_256.rknn           1.67 GB
vae_decoder/model.rknn        0.28 GB
vae_decoder/model_256.rknn    0.16 GB
```

### 6. 上传模型到板子

```powershell
wsl.exe bash -lc "cd /mnt/d/Electronic_Design/Workspace/Projects/RK_Series/Lubancat3 && MODEL_DIR=/home/lzy0x91f/counterfeit_lcm_sd REMOTE_MODEL_DIR=/home/cat/lcm_sd BOARD_USER=cat PASS=2335 bash scripts/wsl_upload_model.sh"
```

### 7. 上传板端脚本

```powershell
D:\Anaconda3\envs\comprehensive\python.exe host\deploy_board.py
```

不要用旧 Dreamshaper 的 `--with-embeds` 覆盖当前缓存。需要缓存时，在板子上重新生成：

```bash
python3 /home/cat/sd_lcm.py --mode fast --save-embeds --prompt "masterpiece, best quality, anime girl"
```

## 验证命令

查看模式：

```bash
python3 /home/cat/sd_lcm.py --list-modes
```

快速测试：

```bash
python3 /home/cat/sd_lcm.py --mode fast --seed 42 --out /home/cat/sd_outputs/counterfeit_fast_test.png --json \
  --prompt "masterpiece, best quality, anime illustration, 1girl, white hair, blue eyes, detailed eyes, soft light"
```

512 测试：

```bash
python3 /home/cat/sd_lcm.py --mode balanced --seed 42 --out /home/cat/sd_outputs/counterfeit_balanced_test.png --json \
  --prompt "masterpiece, best quality, anime illustration, 1girl, white hair, blue eyes, detailed eyes, soft light"
```

本次板端实测结果：

```text
fast:     256x256, 4 step,  total about 24s to 45s
balanced: 512x512, 8 step,  total about 154s
quality:  512x512, 12 step, total about 204s
```

## Dreamshaper 旧模型归档

旧模型已经从板端删除，但 GitHub 通过 Git LFS 保留：

```text
models/dreamshaper_lcm_rknn/
```

第一次克隆后如果要拉旧模型：

```bash
git lfs install
git lfs pull
```

恢复到板子时，把该目录内容上传到 `/home/cat/lcm_sd`，再部署板端脚本即可。

## 常见问题

### 为什么看到 `Query dynamic range failed`？

这是静态 shape RKNN 模型查询动态范围时的 warning。当前模型可以正常 load/init/inference，可以忽略。

### 为什么 prompt 很长时没效果？

CLIP 最多 77 token，实际内容约 75 token，超出的部分会被截断。二次元模型更适合短 prompt。

### 为什么 Counterfeit 的 CFG 这么低？

这是 LCM LoRA 用法。普通 SD 常见 CFG 7 到 8，但 LCM LoRA 常用 1.0 左右。CFG 太高可能导致发灰、过曝或结构变差。

### 为什么 quality 不一定总比 balanced 更好？

LCM 不是步数越多越好。对这套模型，8 到 12 步比较合适。`quality` 更慢，适合最终图；试 prompt 时优先用 `fast` 或 `balanced`。

### 可以换 Anything、MeinaMix、AOM3 吗？

可以。只要是 SD1.5 checkpoint，流程基本相同：

```text
下载 checkpoint -> 融合 lcm-lora-sdv1-5 -> 导出 ONNX -> WSL 编译 RKNN -> 上传 /home/cat/lcm_sd
```

更换模型时重点确认：

```text
1. 是否 SD1.5 架构
2. tokenizer 是否 CLIP 77 token
3. UNet ONNX 是 3 输入还是 4 输入
4. model_info.json 是否正确
```

## 清理记录

已从板子删除的主要内容：

```text
/home/cat/check_external_data
/home/cat/sd_models
/home/cat/onnx_ref
/home/cat/pytorch_ref
/home/cat/.cache/huggingface
/home/cat/.cache/modelscope
/home/cat/.cache/pip
/home/cat/tmp/pip-*
/home/cat/lcm_sd    # 旧 Dreamshaper，已由 Counterfeit 新模型替换
```

不要删除：

```text
/home/cat/miniconda3
/home/cat/.hermes
/home/cat/projects
/home/cat/lcm_sd    # 当前 Counterfeit 模型
```
