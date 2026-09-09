# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""Qwen Image VAE decoder key names: diffusers -> ComfyUI.

Marigold V2 checkpoints carry a fine-tuned VAE decoder under diffusers names
(``decoder.mid_block.resnets.0.conv1.weight``). ComfyUI loads the same VAE
through its WAN implementation, which uses the original Qwen names
(``decoder.middle.0.residual.2.weight``). The two files hold identical
pretrained weights, so this table was derived by matching tensors by value: all
108 decoder-side keys mapped one-to-one, with no ambiguity and no leftovers.
"""

from __future__ import annotations

DIFFUSERS_TO_COMFY_VAE: dict[str, str] = {
    "decoder.conv_in.bias": "decoder.conv1.bias",
    "decoder.conv_in.weight": "decoder.conv1.weight",
    "decoder.conv_out.bias": "decoder.head.2.bias",
    "decoder.conv_out.weight": "decoder.head.2.weight",
    "decoder.mid_block.attentions.0.norm.gamma": "decoder.middle.1.norm.gamma",
    "decoder.mid_block.attentions.0.proj.bias": "decoder.middle.1.proj.bias",
    "decoder.mid_block.attentions.0.proj.weight": "decoder.middle.1.proj.weight",
    "decoder.mid_block.attentions.0.to_qkv.bias": "decoder.middle.1.to_qkv.bias",
    "decoder.mid_block.attentions.0.to_qkv.weight": "decoder.middle.1.to_qkv.weight",
    "decoder.mid_block.resnets.0.conv1.bias": "decoder.middle.0.residual.2.bias",
    "decoder.mid_block.resnets.0.conv1.weight": "decoder.middle.0.residual.2.weight",
    "decoder.mid_block.resnets.0.conv2.bias": "decoder.middle.0.residual.6.bias",
    "decoder.mid_block.resnets.0.conv2.weight": "decoder.middle.0.residual.6.weight",
    "decoder.mid_block.resnets.0.norm1.gamma": "decoder.middle.0.residual.0.gamma",
    "decoder.mid_block.resnets.0.norm2.gamma": "decoder.middle.0.residual.3.gamma",
    "decoder.mid_block.resnets.1.conv1.bias": "decoder.middle.2.residual.2.bias",
    "decoder.mid_block.resnets.1.conv1.weight": "decoder.middle.2.residual.2.weight",
    "decoder.mid_block.resnets.1.conv2.bias": "decoder.middle.2.residual.6.bias",
    "decoder.mid_block.resnets.1.conv2.weight": "decoder.middle.2.residual.6.weight",
    "decoder.mid_block.resnets.1.norm1.gamma": "decoder.middle.2.residual.0.gamma",
    "decoder.mid_block.resnets.1.norm2.gamma": "decoder.middle.2.residual.3.gamma",
    "decoder.norm_out.gamma": "decoder.head.0.gamma",
    "decoder.up_blocks.0.resnets.0.conv1.bias": "decoder.upsamples.0.residual.2.bias",
    "decoder.up_blocks.0.resnets.0.conv1.weight": "decoder.upsamples.0.residual.2.weight",
    "decoder.up_blocks.0.resnets.0.conv2.bias": "decoder.upsamples.0.residual.6.bias",
    "decoder.up_blocks.0.resnets.0.conv2.weight": "decoder.upsamples.0.residual.6.weight",
    "decoder.up_blocks.0.resnets.0.norm1.gamma": "decoder.upsamples.0.residual.0.gamma",
    "decoder.up_blocks.0.resnets.0.norm2.gamma": "decoder.upsamples.0.residual.3.gamma",
    "decoder.up_blocks.0.resnets.1.conv1.bias": "decoder.upsamples.1.residual.2.bias",
    "decoder.up_blocks.0.resnets.1.conv1.weight": "decoder.upsamples.1.residual.2.weight",
    "decoder.up_blocks.0.resnets.1.conv2.bias": "decoder.upsamples.1.residual.6.bias",
    "decoder.up_blocks.0.resnets.1.conv2.weight": "decoder.upsamples.1.residual.6.weight",
    "decoder.up_blocks.0.resnets.1.norm1.gamma": "decoder.upsamples.1.residual.0.gamma",
    "decoder.up_blocks.0.resnets.1.norm2.gamma": "decoder.upsamples.1.residual.3.gamma",
    "decoder.up_blocks.0.resnets.2.conv1.bias": "decoder.upsamples.2.residual.2.bias",
    "decoder.up_blocks.0.resnets.2.conv1.weight": "decoder.upsamples.2.residual.2.weight",
    "decoder.up_blocks.0.resnets.2.conv2.bias": "decoder.upsamples.2.residual.6.bias",
    "decoder.up_blocks.0.resnets.2.conv2.weight": "decoder.upsamples.2.residual.6.weight",
    "decoder.up_blocks.0.resnets.2.norm1.gamma": "decoder.upsamples.2.residual.0.gamma",
    "decoder.up_blocks.0.resnets.2.norm2.gamma": "decoder.upsamples.2.residual.3.gamma",
    "decoder.up_blocks.0.upsamplers.0.resample.1.bias": "decoder.upsamples.3.resample.1.bias",
    "decoder.up_blocks.0.upsamplers.0.resample.1.weight": "decoder.upsamples.3.resample.1.weight",
    "decoder.up_blocks.0.upsamplers.0.time_conv.bias": "decoder.upsamples.3.time_conv.bias",
    "decoder.up_blocks.0.upsamplers.0.time_conv.weight": "decoder.upsamples.3.time_conv.weight",
    "decoder.up_blocks.1.resnets.0.conv1.bias": "decoder.upsamples.4.residual.2.bias",
    "decoder.up_blocks.1.resnets.0.conv1.weight": "decoder.upsamples.4.residual.2.weight",
    "decoder.up_blocks.1.resnets.0.conv2.bias": "decoder.upsamples.4.residual.6.bias",
    "decoder.up_blocks.1.resnets.0.conv2.weight": "decoder.upsamples.4.residual.6.weight",
    "decoder.up_blocks.1.resnets.0.conv_shortcut.bias": "decoder.upsamples.4.shortcut.bias",
    "decoder.up_blocks.1.resnets.0.conv_shortcut.weight": "decoder.upsamples.4.shortcut.weight",
    "decoder.up_blocks.1.resnets.0.norm1.gamma": "decoder.upsamples.4.residual.0.gamma",
    "decoder.up_blocks.1.resnets.0.norm2.gamma": "decoder.upsamples.4.residual.3.gamma",
    "decoder.up_blocks.1.resnets.1.conv1.bias": "decoder.upsamples.5.residual.2.bias",
    "decoder.up_blocks.1.resnets.1.conv1.weight": "decoder.upsamples.5.residual.2.weight",
    "decoder.up_blocks.1.resnets.1.conv2.bias": "decoder.upsamples.5.residual.6.bias",
    "decoder.up_blocks.1.resnets.1.conv2.weight": "decoder.upsamples.5.residual.6.weight",
    "decoder.up_blocks.1.resnets.1.norm1.gamma": "decoder.upsamples.5.residual.0.gamma",
    "decoder.up_blocks.1.resnets.1.norm2.gamma": "decoder.upsamples.5.residual.3.gamma",
    "decoder.up_blocks.1.resnets.2.conv1.bias": "decoder.upsamples.6.residual.2.bias",
    "decoder.up_blocks.1.resnets.2.conv1.weight": "decoder.upsamples.6.residual.2.weight",
    "decoder.up_blocks.1.resnets.2.conv2.bias": "decoder.upsamples.6.residual.6.bias",
    "decoder.up_blocks.1.resnets.2.conv2.weight": "decoder.upsamples.6.residual.6.weight",
    "decoder.up_blocks.1.resnets.2.norm1.gamma": "decoder.upsamples.6.residual.0.gamma",
    "decoder.up_blocks.1.resnets.2.norm2.gamma": "decoder.upsamples.6.residual.3.gamma",
    "decoder.up_blocks.1.upsamplers.0.resample.1.bias": "decoder.upsamples.7.resample.1.bias",
    "decoder.up_blocks.1.upsamplers.0.resample.1.weight": "decoder.upsamples.7.resample.1.weight",
    "decoder.up_blocks.1.upsamplers.0.time_conv.bias": "decoder.upsamples.7.time_conv.bias",
    "decoder.up_blocks.1.upsamplers.0.time_conv.weight": "decoder.upsamples.7.time_conv.weight",
    "decoder.up_blocks.2.resnets.0.conv1.bias": "decoder.upsamples.8.residual.2.bias",
    "decoder.up_blocks.2.resnets.0.conv1.weight": "decoder.upsamples.8.residual.2.weight",
    "decoder.up_blocks.2.resnets.0.conv2.bias": "decoder.upsamples.8.residual.6.bias",
    "decoder.up_blocks.2.resnets.0.conv2.weight": "decoder.upsamples.8.residual.6.weight",
    "decoder.up_blocks.2.resnets.0.norm1.gamma": "decoder.upsamples.8.residual.0.gamma",
    "decoder.up_blocks.2.resnets.0.norm2.gamma": "decoder.upsamples.8.residual.3.gamma",
    "decoder.up_blocks.2.resnets.1.conv1.bias": "decoder.upsamples.9.residual.2.bias",
    "decoder.up_blocks.2.resnets.1.conv1.weight": "decoder.upsamples.9.residual.2.weight",
    "decoder.up_blocks.2.resnets.1.conv2.bias": "decoder.upsamples.9.residual.6.bias",
    "decoder.up_blocks.2.resnets.1.conv2.weight": "decoder.upsamples.9.residual.6.weight",
    "decoder.up_blocks.2.resnets.1.norm1.gamma": "decoder.upsamples.9.residual.0.gamma",
    "decoder.up_blocks.2.resnets.1.norm2.gamma": "decoder.upsamples.9.residual.3.gamma",
    "decoder.up_blocks.2.resnets.2.conv1.bias": "decoder.upsamples.10.residual.2.bias",
    "decoder.up_blocks.2.resnets.2.conv1.weight": "decoder.upsamples.10.residual.2.weight",
    "decoder.up_blocks.2.resnets.2.conv2.bias": "decoder.upsamples.10.residual.6.bias",
    "decoder.up_blocks.2.resnets.2.conv2.weight": "decoder.upsamples.10.residual.6.weight",
    "decoder.up_blocks.2.resnets.2.norm1.gamma": "decoder.upsamples.10.residual.0.gamma",
    "decoder.up_blocks.2.resnets.2.norm2.gamma": "decoder.upsamples.10.residual.3.gamma",
    "decoder.up_blocks.2.upsamplers.0.resample.1.bias": "decoder.upsamples.11.resample.1.bias",
    "decoder.up_blocks.2.upsamplers.0.resample.1.weight": "decoder.upsamples.11.resample.1.weight",
    "decoder.up_blocks.3.resnets.0.conv1.bias": "decoder.upsamples.12.residual.2.bias",
    "decoder.up_blocks.3.resnets.0.conv1.weight": "decoder.upsamples.12.residual.2.weight",
    "decoder.up_blocks.3.resnets.0.conv2.bias": "decoder.upsamples.12.residual.6.bias",
    "decoder.up_blocks.3.resnets.0.conv2.weight": "decoder.upsamples.12.residual.6.weight",
    "decoder.up_blocks.3.resnets.0.norm1.gamma": "decoder.upsamples.12.residual.0.gamma",
    "decoder.up_blocks.3.resnets.0.norm2.gamma": "decoder.upsamples.12.residual.3.gamma",
    "decoder.up_blocks.3.resnets.1.conv1.bias": "decoder.upsamples.13.residual.2.bias",
    "decoder.up_blocks.3.resnets.1.conv1.weight": "decoder.upsamples.13.residual.2.weight",
    "decoder.up_blocks.3.resnets.1.conv2.bias": "decoder.upsamples.13.residual.6.bias",
    "decoder.up_blocks.3.resnets.1.conv2.weight": "decoder.upsamples.13.residual.6.weight",
    "decoder.up_blocks.3.resnets.1.norm1.gamma": "decoder.upsamples.13.residual.0.gamma",
    "decoder.up_blocks.3.resnets.1.norm2.gamma": "decoder.upsamples.13.residual.3.gamma",
    "decoder.up_blocks.3.resnets.2.conv1.bias": "decoder.upsamples.14.residual.2.bias",
    "decoder.up_blocks.3.resnets.2.conv1.weight": "decoder.upsamples.14.residual.2.weight",
    "decoder.up_blocks.3.resnets.2.conv2.bias": "decoder.upsamples.14.residual.6.bias",
    "decoder.up_blocks.3.resnets.2.conv2.weight": "decoder.upsamples.14.residual.6.weight",
    "decoder.up_blocks.3.resnets.2.norm1.gamma": "decoder.upsamples.14.residual.0.gamma",
    "decoder.up_blocks.3.resnets.2.norm2.gamma": "decoder.upsamples.14.residual.3.gamma",
    "post_quant_conv.bias": "conv2.bias",
    "post_quant_conv.weight": "conv2.weight",
}


def to_comfy_vae_keys(state: dict) -> tuple[dict, list[str]]:
    """Rename a diffusers-named VAE state dict for ComfyUI.

    Returns the renamed dict and the keys that had no counterpart.
    """
    renamed = {}
    unknown = []
    for key, value in state.items():
        target = DIFFUSERS_TO_COMFY_VAE.get(key)
        if target is None:
            unknown.append(key)
        else:
            renamed[target] = value
    return renamed, unknown
