# Agent 使用说明

这是给板端 agent 的操作手册。当前模型是 **Counterfeit V3.0 + SD1.5 LCM LoRA**。优先调用统一入口，不要直接改 scheduler 公式。

## 最推荐的命令

```bash
python3 /home/cat/sd_lcm.py --mode balanced --prompt "masterpiece, best quality, anime girl, cat ears" --json
```

返回内容里会有图片路径，图片旁边还有同名 `.json` 元数据。

## 模式选择

| 目标 | 命令 |
|---|---|
| 快速草图 | `python3 /home/cat/sd_lcm.py --mode fast --prompt "..."`
| 默认生成 | `python3 /home/cat/sd_lcm.py --mode balanced --prompt "..."`
| 质量优先 | `python3 /home/cat/sd_lcm.py --mode quality --prompt "..."`
| 高迭代实验 | `python3 /home/cat/sd_lcm.py --mode quality100 --cached-embeds`
| 极高迭代实验 | `python3 /home/cat/sd_lcm.py --mode quality150 --cached-embeds`

当前板端实测大致耗时：

| 模式 | 分辨率 | 步数 | CFG | 耗时 |
|---|---:|---:|---:|---:|
| fast | 256 | 4 | 1.0 | 约 24 到 45 秒 |
| balanced | 512 | 8 | 1.0 | 约 154 秒 |
| quality | 512 | 12 | 1.2 | 约 204 秒 |
| quality100 | 512 | 100 | 1.0 | 估计约 25 分钟 |
| quality150 | 512 | 150 | 1.0 | 估计约 35 分钟以上 |

## 常用参数

```bash
--prompt "正向提示词"
--negative "负向提示词"
--seed 42
--cfg 1.0
--steps 8
--out /home/cat/sd_outputs/name.png
--json
```

## 可调用 API

### 统一入口

```bash
python3 /home/cat/sd_lcm.py --mode fast|balanced|quality|quality100|quality150
```

### 单独 tokenize

```bash
python3 /home/cat/sd_prompt.py "prompt" "negative"
python3 /home/cat/sd_lcm.py --tokens /tmp/sd_tokens.npz --mode fast
```

### 旧接口兼容

```bash
python3 /home/cat/board_lcm_256.py
python3 /home/cat/board_lcm.py
python3 /home/cat/sd_inference.py --mode fast --prompt "..."
```

### 诊断

```bash
python3 /home/cat/board_diag.py
```

## 注意事项

- 不要删除 `/home/cat/lcm_sd`，这是当前模型目录。
- 不要把 LCM scheduler 改成 DDIM 公式。
- `Query dynamic range failed` 是静态 shape 模型 warning，当前可忽略。
- 256 图是预览，不代表最终质量。
- Counterfeit LCM LoRA 的 CFG 通常用 1.0 到 1.5，不要按普通 SD 直接拉到 7 或 8。
- prompt 最多 77 token，实际内容约 75 token，太长会被截断。
- `sd_lcm.py` 的 JSON 元数据会返回 `prompt_token_count`、`prompt_truncated` 等字段。
- `--cached-embeds` 只适合复用固定 prompt。先读 `/home/cat/pos_emb.npy.json` 确认缓存内容。
- 质量模式较慢，默认 512x512、12 step、CFG 1.2。试 prompt 时优先用 fast 或 balanced。
- `quality100` 和 `quality150` 是实验模式。LCM 通常不需要这么多步；只有用户明确要求高迭代时再用。
