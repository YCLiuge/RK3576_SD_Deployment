# Agent 使用说明

这是给板端 agent 的操作手册。优先调用统一入口，不要直接改 scheduler 公式。

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

## 常用参数

```bash
--prompt "正向提示词"
--negative "负向提示词"
--seed 42
--cfg 6.0
--steps 8
--out /home/cat/sd_outputs/name.png
--json
```

## 可调用 API

### 统一入口

```bash
python3 /home/cat/sd_lcm.py --mode fast|balanced|quality
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
- 质量模式较慢，默认 512x512、8 step、CFG 6.0。

