# RK3576 Stable Diffusion LCM 部署教程

本项目记录并整理了在 **Lubancat3 / RK3576** 上用 **RKNPU** 部署 Stable Diffusion LCM 的完整流程。当前已经跑通 **Dreamshaper V7 LCM**，支持 256x256 和 512x512 生成，并提供统一入口给人和 agent 调用。

> 编译好的 Dreamshaper LCM RKNN 模型已通过 **Git LFS** 备份在 `models/dreamshaper_lcm_rknn/`。普通 git clone 后若没有拉到大文件，请运行 `git lfs pull`。后续如果改用 Counterfeit V3.0，可以安全删除板端 Dreamshaper 模型目录。

## 当前成果

| 项目 | 状态 |
|---|---|
| 板子 | Lubancat3 / RK3576 / 8GB RAM |
| 模型 | Dreamshaper V7 LCM ONNX -> RKNN |
| 运行库 | rknn-toolkit-lite2 2.3.2, librknnrt 2.3.2 |
| 快速模式 | 256x256, 4 step, 可用 |
| 均衡模式 | 512x512, 4 step, 可用 |
| 高质量模式 | 512x512, 100 step, 可用但很慢 |
| 文本输入 | 支持动态 prompt，也支持预计算 embedding |
| 板端空间 | 已清理旧模型和缓存，根分区从约 78% 降到约 49% |

样图（示意图；当前 `quality` 默认已改为 100 step，实际最终图通常应优于早期 8 step 样图）：

![fast sample](assets/sample_fast.png)
![balanced sample](assets/sample_balanced.png)
![quality sample](assets/sample_quality.png)

## 一句话使用

在板子上运行：

```bash
python3 /home/cat/sd_lcm.py --mode fast --prompt "masterpiece, best quality, cat girl, white hair, blue eyes"
```

输出默认保存在：

```text
/home/cat/sd_outputs/
```

## 三种模式

| 模式 | 命令 | 分辨率 | 步数 | 默认 CFG | 适合场景 |
|---|---|---:|---:|---:|---|
| 快速 | `--mode fast` | 256x256 | 4 | 7.0 | 快速看构图、简单图、agent 草稿 |
| 均衡 | `--mode balanced` | 512x512 | 4 | 7.5 | 默认推荐，速度和效果平衡 |
| 高质量 | `--mode quality` | 512x512 | 100 | 6.0 | 最终二次元图，适合愿意等待十几分钟以上 |

示例：

```bash
# 快速预览
python3 /home/cat/sd_lcm.py --mode fast --prompt "masterpiece, best quality, anime girl, cat ears"

# 均衡默认
python3 /home/cat/sd_lcm.py --mode balanced --prompt "masterpiece, best quality, 1girl, white hair, blue eyes"

# 高质量，指定 seed 和输出路径。100 step 会明显更慢。
python3 /home/cat/sd_lcm.py --mode quality --seed 123 --out /home/cat/sd_outputs/quality_123.png \
  --prompt "masterpiece, best quality, anime illustration, cat girl, detailed eyes, soft light"
```

## 项目结构

```text
.
├─ board/                  # 上传到板子 /home/cat 后直接运行
│  ├─ sd_lcm.py            # 统一入口，推荐 agent 调用
│  ├─ sd_prompt.py         # 只做 prompt tokenize，输出 /tmp/sd_tokens.npz
│  ├─ sd_inference.py      # 兼容旧接口的 wrapper
│  ├─ board_lcm.py         # 兼容旧 512 入口
│  ├─ board_lcm_256.py     # 兼容旧 256 入口
│  ├─ board_gen_emb.py     # 板端生成 pos_emb.npy / neg_emb.npy
│  └─ board_diag.py        # 板端本地诊断
├─ host/                   # 电脑端辅助脚本
│  ├─ deploy_board.py      # 上传 board/ 脚本到板子
│  ├─ remote_board_diag.py # 从电脑 SSH 运行板端诊断
│  ├─ laptop_lcm.py        # 512 ONNX golden reference
│  ├─ laptop_lcm_256.py    # 256 ONNX golden reference
│  └─ gen_ds_emb.py        # 电脑端用 ONNX text_encoder 生成 embedding
├─ scripts/                # WSL 下载、编译、上传 RKNN 的脚本
├─ configs/                # scheduler_config.json 参考配置
├─ embeds/                 # 小体积预计算 embedding，可选
├─ assets/                 # README 样图
└─ docs/                   # 补充文档
```

## 板端文件布局

板子上保留的核心目录如下：

```text
/home/cat/
├─ sd_lcm.py
├─ sd_prompt.py
├─ sd_inference.py
├─ board_lcm.py
├─ board_lcm_256.py
├─ board_gen_emb.py
├─ board_diag.py
├─ pos_emb.npy             # 可选：预计算正向 embedding
├─ neg_emb.npy             # 可选：预计算负向 embedding
├─ sd_outputs/             # 输出图片和 json 元数据
└─ lcm_sd/
   ├─ text_encoder/model.rknn
   ├─ unet/model_256.rknn
   ├─ unet/model.rknn
   ├─ vae_decoder/model_256.rknn
   ├─ vae_decoder/model.rknn
   ├─ scheduler/scheduler_config.json
   └─ tokenizer/{merges.txt,vocab.json,...}
```

必须有的是 `/home/cat/lcm_sd` 里的模型和 tokenizer。`pos_emb.npy` / `neg_emb.npy` 只是加速默认 prompt，不是必须。

GitHub LFS 备份位置：

```text
models/dreamshaper_lcm_rknn/
├─ text_encoder/model.rknn
├─ unet/model_256.rknn
├─ unet/model.rknn
├─ vae_decoder/model_256.rknn
├─ vae_decoder/model.rknn
├─ scheduler/scheduler_config.json
└─ tokenizer/
```

## 重要参数

`sd_lcm.py` 常用参数：

```bash
python3 /home/cat/sd_lcm.py --help
```

| 参数 | 说明 |
|---|---|
| `--mode fast|balanced|quality` | 选择预设模式 |
| `--prompt "..."` | 正向提示词 |
| `--negative "..."` | 负向提示词 |
| `--seed 42` | 随机种子，同配置下可复现 |
| `--cfg 7.0` | 提示词引导强度，太高容易过曝/崩脸 |
| `--steps 4` | LCM 推理步数，越多越慢；实测该模型 100 步以上质量更稳 |
| `--resolution 256|512` | 覆盖模式默认分辨率 |
| `--out path.png` | 指定输出图片 |
| `--cached-embeds` | 使用 `/home/cat/pos_emb.npy` 和 `neg_emb.npy` |
| `--save-embeds` | 动态 prompt 后保存 embedding |
| `--tokens /tmp/sd_tokens.npz` | 使用 `sd_prompt.py` 预先 tokenize 的结果 |
| `--json` | 额外输出一行 JSON 元数据，方便 agent 解析 |

输出图片旁边会生成一个 `.json` 文件，记录 prompt、模式、seed、耗时、timesteps 等信息。

## Agent 推荐调用方式

给 agent 最简单的主入口：

```bash
python3 /home/cat/sd_lcm.py --mode balanced --prompt "masterpiece, best quality, cat girl" --json
```

如果 agent 想分两步控制：

```bash
python3 /home/cat/sd_prompt.py "masterpiece, best quality, cat girl" "low quality, bad anatomy"
python3 /home/cat/sd_lcm.py --mode fast --tokens /tmp/sd_tokens.npz --json
```

如果只想最快复用默认 embedding：

```bash
python3 /home/cat/sd_lcm.py --mode fast --cached-embeds --json
```

更多 agent 说明见 [AGENT.md](AGENT.md)。

## 部署流程

### 1. 准备板端 Python 环境

板子上需要：

```bash
python3 -c "import numpy; from PIL import Image; from tokenizers import Tokenizer; from rknnlite.api import RKNNLite; print('ok')"
```

当前验证过：

```text
numpy 1.26.4
Pillow OK
tokenizers OK
rknn-toolkit-lite2 2.3.2
librknnrt 2.3.2
NPU driver 0.9.8
```

`rknn_server` 仍是 2.1.0，但本项目板端本地推理主要依赖 `librknnrt`，不依赖 `rknn_server`。`rknn_server` 主要用于电脑远程 accuracy_analysis。

### 2. 下载 ONNX 模型

模型来源：

```text
TheyCallMeHex/LCM-Dreamshaper-V7-ONNX
```

WSL 脚本：

```bash
bash scripts/wsl_lcm_dl.sh
```

下载后应该得到：

```text
lcm_sd/
├─ text_encoder/model.onnx
├─ unet/model.onnx
├─ unet/model.onnx_data
├─ vae_decoder/model.onnx
├─ scheduler/scheduler_config.json
└─ tokenizer/
```

### 3. 编译 RKNN

512 模型：

```bash
bash scripts/wsl_lcm_compile.sh
```

256 模型：

```bash
bash scripts/wsl_compile_256.sh
```

目标平台必须是：

```python
target_platform='rk3576'
```

编译后板端需要这些文件：

```text
/home/cat/lcm_sd/text_encoder/model.rknn
/home/cat/lcm_sd/unet/model.rknn
/home/cat/lcm_sd/unet/model_256.rknn
/home/cat/lcm_sd/vae_decoder/model.rknn
/home/cat/lcm_sd/vae_decoder/model_256.rknn
```

### 4. 上传模型和脚本

上传 RKNN 模型：

```bash
bash scripts/wsl_upload_lcm.sh
bash scripts/wsl_up256.sh
```

上传 Python 脚本：

```powershell
D:\Anaconda3\envs\comprehensive\python.exe host\deploy_board.py --with-embeds
```

如果板子 IP 不是 `10.138.103.190`：

```powershell
D:\Anaconda3\envs\comprehensive\python.exe host\deploy_board.py --host 你的IP --with-embeds
```

### 5. 板端诊断

板端本地：

```bash
python3 /home/cat/board_diag.py
```

电脑端远程：

```powershell
D:\Anaconda3\envs\comprehensive\python.exe host\remote_board_diag.py
```

## 方法论：这次踩坑的真正原因

最开始输出是纯橙色块，很容易误判为 RK3576 NPU 精度不足。实际排查发现：

1. 板端 NPU 输出和本机 ONNX 输出几乎一样，说明 NPU 没把这条链路算坏。
2. 旧代码把 LCM 当 DDIM/sigma scheduler 用了。
3. `LCMScheduler.scale_model_input()` 对 LCM 是原样返回，旧代码错误地做了 `latent / sqrt(1 + sigma^2)`。
4. 真正的 LCM step 需要 `c_skip/c_out` boundary condition，并且非最后 step 注入随机噪声。
5. LCM 4 step timesteps 应为 `[999, 759, 499, 259]`，不是 `[751, 501, 251, 1]`。
6. ONNX `timestep` 输入类型是 `int64`。

修复这些以后，ONNX 与 RK3576 NPU 输出一致，512 图像相关性约 0.998。

## 常见问题

### 为什么 fast 总耗时比 UNet 6 秒更长？

日志里的 `unet_time_sec` 只统计 UNet 推理循环。总耗时还包括 RKNN 模型加载、初始化、text_encoder 和 VAE。首次运行通常更慢，连续运行会有波动。

### 为什么 256 图比较糊？

256x256 是预览模式，目标是快。想要更好的二次元图，请用：

```bash
python3 /home/cat/sd_lcm.py --mode quality --prompt "更完整的提示词"
```

当前 `quality` 默认是 512x512、100 step、CFG 6.0，速度会比 `balanced` 慢很多。它是“最终图”档，不适合频繁试 prompt。

### 为什么会看到 `Query dynamic range failed`？

这是静态 shape RKNN 模型查询动态范围时的 warning。当前模型能正常 load/init/inference，可以忽略。

### GitHub 上的模型文件怎么看不到或拉不全？

编译好的 Dreamshaper LCM RKNN 模型在 Git LFS 中。第一次克隆后请运行：

```bash
git lfs install
git lfs pull
```

如果不想下载旧 Dreamshaper 模型，也可以只用 `scripts/` 重新下载和编译新模型。

### 如果换 prompt 后效果很差怎么办？

优先尝试：

```bash
--mode balanced          # 先试构图
--mode quality           # 最终图，100 step
--seed 其他数字
--cfg 5.0 到 7.5
--steps 100 到 150       # 愿意更久时可以覆盖 quality 默认
```

二次元 prompt 建议包含质量词、主体、发色/眼睛/光照/构图，例如：

```text
masterpiece, best quality, anime illustration, 1girl, cat ears, white hair, blue eyes, detailed eyes, soft light
```

负向词可以用：

```text
worst quality, low quality, lowres, bad anatomy, bad hands, text, watermark, blurry
```

## 板端清理记录

已删除：

```text
/home/cat/check_external_data        # 精度分析中间数据，约 3.3G
/home/cat/sd_models                  # 旧 Counterfeit/TAESD 链路，约 3.7G
/home/cat/onnx_ref, pytorch_ref      # 旧验证数据
/home/cat/.cache/huggingface         # 缓存，约 4.9G
/home/cat/.cache/modelscope          # 缓存，约 1.9G
/home/cat/.cache/pip                 # 缓存，约 2.2G
/home/cat/tmp/pip-*                  # pip 临时目录
```

未删除：

```text
/home/cat/lcm_sd                     # 当前部署模型，必须保留
/home/cat/miniconda3                 # Python/conda 环境
/home/cat/.hermes, /home/cat/projects, Desktop, Downloads 等非本项目内容
```

清理后根分区约：

```text
59G 总容量，28G 已用，29G 可用，约 49%
```
