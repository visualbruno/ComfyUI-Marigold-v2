# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""Static description of the Marigold V2 model family.

Mirrors ``scripts/infer.py`` and ``evaluation/config/inference_depth.yaml`` of the
upstream repository (https://github.com/huawei-bayerlab/marigold-v2).
"""

from __future__ import annotations

from dataclasses import dataclass

QWEN_REPO = "Qwen/Qwen-Image-Edit-2509"
# Only the DiT and the VAE are ever loaded; the 7B text encoder is replaced by
# the precomputed prompt embeddings shipped with the Marigold V2 checkpoints.
QWEN_ALLOW_PATTERNS = ["model_index.json", "transformer/*", "vae/*"]

MARIGOLD_REPO = "huawei-bayerlab/marigold-v2-0"
EMBED_SUBDIR = "qwen_text_embeddings"

QWEN_DIR_NAME = "Qwen-Image-Edit-2509"
MARIGOLD_DIR_NAME = "Marigold-V2"
GGUF_DIR_NAME = "gguf"
TRAINABLES_FILENAME = "trainables.safetensors"

# The VAE, the DiT config, and the pipeline index are needed whatever the
# transformer weights come from; together they are about 250 MB.
QWEN_SMALL_PATTERNS = ["model_index.json", "transformer/config.json", "vae/*"]

# GGUF builds of the same 2509 DiT. The architecture is identical, only the
# quantization differs, so the Marigold LoRA loads on top unchanged.
GGUF_REPO = "QuantStack/Qwen-Image-Edit-2509-GGUF"
GGUF_FILE_TEMPLATE = "Qwen-Image-Edit-2509-{quant}.gguf"
GGUF_QUANTS = {  # quant -> approximate download size in GB
    "Q8_0": 22,
    "Q6_K": 17,
    "Q5_K_M": 15,
    "Q4_K_M": 13,
    "Q4_0": 12,
    "Q3_K_M": 10,
}

BF16_BACKBONE = "Qwen-Image-Edit-2509 bf16 (41 GB download)"


def gguf_download_label(quant: str) -> str:
    return f"Qwen-Image-Edit-2509 GGUF {quant} ({GGUF_QUANTS[quant]} GB download)"


GGUF_DOWNLOAD_LABELS = {gguf_download_label(q): q for q in GGUF_QUANTS}

# Fixed rectified-flow timestep the models were fine-tuned at (single step).
TIMESTEP = 499.0

# Per-channel statistics of the Qwen Image VAE latent space, from the VAE
# config. Needed verbatim on the ComfyUI path, which has no diffusers config to
# read them from.
LATENTS_MEAN = [
    -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
    0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921,
]
LATENTS_STD = [
    2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
    3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160,
]

# LoRA configuration shared by every released checkpoint.
LORA_RANK = 128
LORA_ALPHA = 128
LORA_DROPOUT = 0.0
LORA_INIT = "gaussian"
LORA_TARGET_MODULES = [
    "img_in",
    "txt_in",
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.add_q_proj",
    "attn.to_add_out",
    "norm.linear",
    "a_to_out",
    "b_to_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
    "ff_a.0",
    "ff_a.2",
    "ff_b.0",
    "ff_b.2",
    "norm_out.linear",
    "proj_out",
]

QUANT_SKIP_MODULES = ["transformer_blocks.0.img_mod"]
QUANT_DEQUANTIZE_MODULES = ["proj_out"]

# Prompt embeddings replace the text encoder, one set per modality.
EMBED_PREFIX = {
    "depth": "qwen_edit_2509_qwen_depth_realimg512",
    "normals": "qwen_edit_2509_qwen_normals_dummy512",
    "albedo": "qwen_edit_2509_qwen_albedo_rgb_dummy512",
}

MODALITIES = ["depth", "normals", "albedo"]


@dataclass(frozen=True)
class CheckpointSpec:
    """One released checkpoint plus everything needed to run and read it."""

    modality: str
    variant: str
    subfolder: str
    embed_prefix: str
    # Depth only: True when the predicted values grow with distance (log and
    # linear depth), False for disparity, which decreases with distance.
    far_is_high: bool = True

    @property
    def label(self) -> str:
        return f"{self.modality}/{self.variant}" if self.variant else self.modality


# Depth checkpoints differ in how depth is parameterized before the VAE.
DEPTH_VARIANTS = {
    "Log-stage2": True,
    "Log-stage1": True,
    "Log-layered": True,
    "Uniform-base": True,
    "Uniform-layered": True,
    "Disparity-base": False,
    "Disparity-layered": False,
}
DEFAULT_DEPTH_VARIANT = "Log-stage2"


def get_spec(modality: str, depth_variant: str = DEFAULT_DEPTH_VARIANT) -> CheckpointSpec:
    if modality not in MODALITIES:
        raise ValueError(f"Unknown modality '{modality}'; expected one of {MODALITIES}")
    if modality == "depth":
        if depth_variant not in DEPTH_VARIANTS:
            raise ValueError(
                f"Unknown depth checkpoint '{depth_variant}'; "
                f"expected one of {sorted(DEPTH_VARIANTS)}"
            )
        return CheckpointSpec(
            modality="depth",
            variant=depth_variant,
            subfolder=f"depth/{depth_variant}",
            embed_prefix=EMBED_PREFIX["depth"],
            far_is_high=DEPTH_VARIANTS[depth_variant],
        )
    return CheckpointSpec(
        modality=modality,
        variant="",
        subfolder=modality,
        embed_prefix=EMBED_PREFIX[modality],
    )


# Depth parameterizations, for checkpoints named by the user rather than picked
# from the release table (the ComfyUI LoRA path).
DEPTH_PARAMETERIZATIONS = {
    "log_or_linear": True,   # Log-* and Uniform-*: values grow with distance
    "disparity": False,      # Disparity-*: values shrink with distance
}


def spec_for_file(modality: str, parameterization: str = "log_or_linear") -> CheckpointSpec:
    """A spec for a checkpoint loaded from an arbitrary file."""
    if modality not in MODALITIES:
        raise ValueError(f"Unknown modality '{modality}'; expected one of {MODALITIES}")
    if parameterization not in DEPTH_PARAMETERIZATIONS:
        raise ValueError(
            f"Unknown depth parameterization '{parameterization}'; "
            f"expected one of {sorted(DEPTH_PARAMETERIZATIONS)}"
        )
    return CheckpointSpec(
        modality=modality,
        variant="",
        subfolder=modality,
        embed_prefix=EMBED_PREFIX[modality],
        far_is_high=DEPTH_PARAMETERIZATIONS[parameterization],
    )
