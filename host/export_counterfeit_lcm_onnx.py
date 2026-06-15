#!/usr/bin/env python3
"""Export Counterfeit V3.0 + SD1.5 LCM LoRA to ONNX for RKNN compilation."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
from diffusers import LCMScheduler, StableDiffusionPipeline
from diffusers.models.attention_processor import AttnProcessor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / "models" / "checkpoints" / "counterfeit_v30" / "Counterfeit-V3.0_fix_fp16.safetensors"
DEFAULT_LORA = ROOT / "models" / "checkpoints" / "lcm_lora_sdv15" / "pytorch_lora_weights.safetensors"
DEFAULT_ORIGINAL_CONFIG = ROOT / "configs" / "sd15_v1_inference.yaml"
DEFAULT_OUT = ROOT / "models" / "counterfeit_lcm_onnx"


class TextEncoderExport(torch.nn.Module):
    def __init__(self, text_encoder):
        super().__init__()
        self.text_encoder = text_encoder

    def forward(self, input_ids):
        return self.text_encoder(input_ids, return_dict=False)[0]


class UNetExport(torch.nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]


class VAEDecoderExport(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latent_sample):
        return self.vae.decode(latent_sample, return_dict=False)[0]


def export_model(model, inputs, output_path, input_names, output_names, dynamic_axes, opset):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with torch.no_grad():
        torch.onnx.export(
            model,
            inputs,
            str(output_path),
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            external_data=True,
        )
    print(f"exported {output_path}")


def save_model_info(args, pipe):
    info = {
        "name": "Counterfeit V3.0 + LCM LoRA SD1.5",
        "base_model": "gsdf/Counterfeit-V3.0",
        "base_file": Path(args.checkpoint).name,
        "lora": "latent-consistency/lcm-lora-sdv1-5",
        "lora_file": Path(args.lora).name,
        "unet_inputs": ["sample", "timestep", "encoder_hidden_states"],
        "unet_input_count": 3,
        "tokenizer_max_length": int(pipe.tokenizer.model_max_length),
        "text_hidden_size": int(pipe.text_encoder.config.hidden_size),
        "latent_channels": int(pipe.unet.config.in_channels),
        "vae_scaling_factor": float(pipe.vae.config.scaling_factor),
        "default_prompt": "masterpiece, best quality, anime illustration, 1girl, detailed eyes, soft light",
        "default_negative": "worst quality, low quality, lowres, bad anatomy, bad hands, text, watermark, blurry",
        "modes": {
            "fast": {
                "resolution": 256,
                "steps": 4,
                "cfg": 1.0,
                "description": "Fast 256x256 Counterfeit LCM preview.",
            },
            "balanced": {
                "resolution": 512,
                "steps": 8,
                "cfg": 1.0,
                "description": "Balanced 512x512 Counterfeit LCM generation.",
            },
            "quality": {
                "resolution": 512,
                "steps": 12,
                "cfg": 1.2,
                "description": "Slower 512x512 Counterfeit LCM final image.",
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
        },
    }
    path = Path(args.out_dir) / "model_info.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"wrote {path}")


def load_pipeline(args, dtype, device):
    pipe = StableDiffusionPipeline.from_single_file(
        args.checkpoint,
        original_config=args.original_config,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(args.lora)
    pipe.fuse_lora(lora_scale=args.lora_scale)
    pipe.unload_lora_weights()
    pipe.unet.set_attn_processor(AttnProcessor())
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--lora", default=str(DEFAULT_LORA))
    parser.add_argument("--original-config", default=str(DEFAULT_ORIGINAL_CONFIG))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16" if torch.cuda.is_available() else "fp32")
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    device = torch.device(args.device)
    print(f"loading pipeline on {device} ({args.dtype})")
    pipe = load_pipeline(args, dtype, device)

    pipe.tokenizer.save_pretrained(out_dir / "tokenizer")
    pipe.scheduler.save_config(out_dir / "scheduler")
    save_model_info(args, pipe)

    input_ids = torch.ones((1, 77), dtype=torch.long, device=device)
    export_model(
        TextEncoderExport(pipe.text_encoder),
        (input_ids,),
        out_dir / "text_encoder" / "model.onnx",
        ["input_ids"],
        ["last_hidden_state"],
        {"input_ids": {0: "batch"}, "last_hidden_state": {0: "batch"}},
        args.opset,
    )

    latent = torch.randn((2, 4, 64, 64), dtype=dtype, device=device)
    timestep = torch.tensor([999], dtype=torch.long, device=device)
    hidden = torch.randn((2, 77, pipe.text_encoder.config.hidden_size), dtype=dtype, device=device)
    export_model(
        UNetExport(pipe.unet),
        (latent, timestep, hidden),
        out_dir / "unet" / "model.onnx",
        ["sample", "timestep", "encoder_hidden_states"],
        ["out_sample"],
        {
            "sample": {0: "batch", 2: "latent_height", 3: "latent_width"},
            "encoder_hidden_states": {0: "batch"},
            "out_sample": {0: "batch", 2: "latent_height", 3: "latent_width"},
        },
        args.opset,
    )

    vae_latent = torch.randn((1, 4, 64, 64), dtype=dtype, device=device)
    export_model(
        VAEDecoderExport(pipe.vae),
        (vae_latent,),
        out_dir / "vae_decoder" / "model.onnx",
        ["latent_sample"],
        ["sample"],
        {
            "latent_sample": {0: "batch", 2: "latent_height", 3: "latent_width"},
            "sample": {0: "batch", 2: "image_height", 3: "image_width"},
        },
        args.opset,
    )

    print("done")


if __name__ == "__main__":
    main()
