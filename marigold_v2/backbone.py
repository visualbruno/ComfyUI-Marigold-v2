# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""Resolving the loader's ``backbone`` choice into transformer weights.

Marigold V2 was fine-tuned on Qwen-Image-Edit-2509 with the DiT quantized to
NF4, so the official bf16 repo reproduces the paper exactly. A GGUF build of
the same 2509 model has identical architecture and tensor names - only the
quantization differs - so the LoRA loads on top unchanged, for a third of the
download. A GGUF of a *different* Qwen-Image-Edit release (2511 and later) also
loads, but the adapter was never trained against those weights.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from . import download, paths
from .constants import BF16_BACKBONE, GGUF_DOWNLOAD_LABELS, gguf_download_label


@dataclass(frozen=True)
class TransformerSource:
    kind: str  # "diffusers" or "gguf"
    path: str  # a model directory, or a .gguf file
    label: str


def options() -> list[str]:
    """Widget choices: the official download, GGUF downloads, then local GGUFs."""
    return (
        [BF16_BACKBONE]
        + [gguf_download_label(quant) for quant in GGUF_DOWNLOAD_LABELS.values()]
        + list(paths.list_local_gguf())
    )


def resolve(label: str, auto_download: bool) -> tuple[str, TransformerSource]:
    """Return ``(qwen_dir, source)`` for one ``backbone`` widget value.

    ``qwen_dir`` always holds the VAE and the DiT config; the source says where
    the transformer weights themselves come from.
    """
    qwen_dir = download.ensure_vae_and_config(auto_download)

    if label == BF16_BACKBONE:
        path = download.ensure_bf16_transformer(auto_download)
        return qwen_dir, TransformerSource("diffusers", path, label)

    quant = GGUF_DOWNLOAD_LABELS.get(label)
    if quant is not None:
        path = download.ensure_gguf(quant, auto_download)
        return qwen_dir, TransformerSource("gguf", path, label)

    local = paths.list_local_gguf().get(label)
    if local is None:
        if os.path.isfile(label) and label.lower().endswith(".gguf"):
            local = label  # an explicit path, e.g. from an API workflow
        else:
            raise download.ModelsMissing(
                f"Unknown backbone '{label}'. Pick one of the download options, "
                "or put a .gguf in ComfyUI/models/diffusion_models and reload "
                "the node."
            )
    return qwen_dir, TransformerSource("gguf", local, label)


def warn_if_not_2509(source: TransformerSource) -> str | None:
    """A caution for GGUFs that are clearly not the 2509 release."""
    if source.kind != "gguf":
        return None
    name = os.path.basename(source.path).lower()
    if "2509" in name:
        return None
    return (
        f"'{os.path.basename(source.path)}' does not look like a "
        "Qwen-Image-Edit-2509 build. The Marigold V2 adapter was trained "
        "against 2509 weights. Later releases share the architecture and do "
        "produce coherent predictions, but they are not the configuration the "
        "paper measured."
    )
