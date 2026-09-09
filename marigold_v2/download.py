# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""Hugging Face downloads for the Marigold V2 nodes.

Three pieces, fetched independently so nobody pays for what they do not use:

* the Qwen VAE plus the DiT config (~250 MB) - always needed;
* the DiT weights - either the official bf16 repo (~41 GB, quantized to NF4 at
  load, exactly what the checkpoints were trained against) or a GGUF build of
  the same 2509 model (10-22 GB), or a GGUF already on disk;
* one ~1.8 GB ``trainables.safetensors`` per modality plus the tiny precomputed
  prompt embeddings, from ``huawei-bayerlab/marigold-v2-0``.
"""

from __future__ import annotations

import logging
import os
import time

from . import paths
from .constants import (
    EMBED_SUBDIR,
    GGUF_FILE_TEMPLATE,
    GGUF_REPO,
    MARIGOLD_REPO,
    QWEN_ALLOW_PATTERNS,
    QWEN_REPO,
    QWEN_SMALL_PATTERNS,
    TRAINABLES_FILENAME,
    CheckpointSpec,
)

log = logging.getLogger("ComfyUI-Marigold-v2")

ATTEMPTS = 4
WORKERS = 4

# Finite socket timeouts so a stalled connection fails and gets retried.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")


class ModelsMissing(RuntimeError):
    """Raised when weights are absent and automatic download is disabled."""


def _retry(what: str, fn):
    last: Exception | None = None
    for attempt in range(1, ATTEMPTS + 1):
        try:
            return fn()
        except Exception as exc:  # network flakiness; resume on the next pass
            last = exc
            if attempt < ATTEMPTS:
                delay = 2**attempt
                log.warning(
                    "[Marigold V2] %s failed (%r); resuming in %ss (attempt %d/%d)",
                    what,
                    exc,
                    delay,
                    attempt,
                    ATTEMPTS,
                )
                time.sleep(delay)
    raise RuntimeError(
        f"Failed to download {what} after {ATTEMPTS} attempts. Re-run to resume, "
        "or fetch the files manually (see the node pack README)."
    ) from last


def _snapshot(repo: str, target: str, patterns: list[str]) -> None:
    from huggingface_hub import snapshot_download

    os.makedirs(target, exist_ok=True)
    _retry(
        repo,
        lambda: snapshot_download(
            repo_id=repo,
            repo_type="model",
            local_dir=target,
            allow_patterns=patterns,
            max_workers=WORKERS,
        ),
    )


def ensure_vae_and_config(auto_download: bool) -> str:
    """The Qwen VAE and DiT config, needed whatever the transformer source is."""
    target = paths.qwen_dir()
    have = all(
        os.path.isfile(os.path.join(target, *parts))
        for parts in (("vae", "config.json"), ("transformer", "config.json"))
    ) and os.path.isdir(os.path.join(target, "vae"))
    if have:
        return target

    if not auto_download:
        raise ModelsMissing(
            f"The Qwen VAE and DiT config are not in {target}. Enable "
            f"'auto_download' on the loader node, or copy the 'vae' folder and "
            f"'transformer/config.json' of {QWEN_REPO} into that directory."
        )
    log.info("[Marigold V2] Downloading the Qwen VAE and DiT config (~250 MB)")
    _snapshot(QWEN_REPO, target, QWEN_SMALL_PATTERNS)
    return target


def ensure_bf16_transformer(auto_download: bool) -> str:
    """The official bf16 DiT weights, quantized to NF4 when they are loaded."""
    target = paths.qwen_dir()
    index = os.path.join(target, "transformer", "diffusion_pytorch_model.safetensors.index.json")
    if os.path.isfile(index):
        return target

    if not auto_download:
        raise ModelsMissing(
            f"The Qwen-Image-Edit-2509 transformer weights are not in {target}. "
            "Enable 'auto_download', pick a GGUF backbone instead, or copy the "
            f"'transformer' folder of {QWEN_REPO} into that directory."
        )
    log.info("[Marigold V2] Downloading %s transformer + vae (~41 GB)", QWEN_REPO)
    _snapshot(QWEN_REPO, target, QWEN_ALLOW_PATTERNS)
    return target


def ensure_gguf(quant: str, auto_download: bool) -> str:
    """One GGUF build of the 2509 DiT, downloaded if it is not already local."""
    filename = GGUF_FILE_TEMPLATE.format(quant=quant)
    local = paths.list_local_gguf().get(filename)
    if local is not None:
        return local

    target = paths.gguf_dir(create=auto_download)
    destination = os.path.join(target, filename)
    if not auto_download:
        raise ModelsMissing(
            f"{filename} was not found. Enable 'auto_download' on the loader "
            f"node, or download it from {GGUF_REPO} into {target} (or any "
            "ComfyUI diffusion_models folder)."
        )

    from huggingface_hub import hf_hub_download

    log.info("[Marigold V2] Downloading %s/%s", GGUF_REPO, filename)
    _retry(
        f"{GGUF_REPO}/{filename}",
        lambda: hf_hub_download(
            repo_id=GGUF_REPO, filename=filename, repo_type="model", local_dir=target
        ),
    )
    return destination


def _ensure_file(relative: str, target_dir: str, auto_download: bool, note: str) -> str:
    local = os.path.join(target_dir, relative.replace("/", os.sep))
    if os.path.isfile(local):
        return local
    if not auto_download:
        raise ModelsMissing(
            f"{note} not found at {local}. Enable 'auto_download' on the loader "
            f"node, or download '{relative}' from {MARIGOLD_REPO} into {target_dir}."
        )

    from huggingface_hub import hf_hub_download

    os.makedirs(target_dir, exist_ok=True)
    log.info("[Marigold V2] Downloading %s/%s", MARIGOLD_REPO, relative)
    _retry(
        f"{MARIGOLD_REPO}/{relative}",
        lambda: hf_hub_download(
            repo_id=MARIGOLD_REPO,
            filename=relative,
            repo_type="model",
            local_dir=target_dir,
        ),
    )
    return local


def ensure_prompt_embeddings(spec: CheckpointSpec, auto_download: bool) -> tuple[str, str]:
    """Just the prompt embeddings for one modality (a few MB).

    These stand in for the 7B text encoder and cannot be derived from anything
    local, so even the bring-your-own-weights path needs them.
    """
    target = paths.marigold_dir(create=auto_download)
    embeds = _ensure_file(
        f"{EMBED_SUBDIR}/{spec.embed_prefix}_prompt_embeds.pt",
        target,
        auto_download,
        f"{spec.modality} prompt embeddings",
    )
    mask = _ensure_file(
        f"{EMBED_SUBDIR}/{spec.embed_prefix}_prompt_mask.pt",
        target,
        auto_download,
        f"{spec.modality} prompt mask",
    )
    return embeds, mask


def ensure_checkpoint(spec: CheckpointSpec, auto_download: bool) -> tuple[str, str, str]:
    """Fetch one checkpoint and its prompt embeddings.

    Returns ``(trainables_path, prompt_embeds_path, prompt_mask_path)``.
    """
    target = paths.marigold_dir(create=auto_download)
    trainables = _ensure_file(
        f"{spec.subfolder}/{TRAINABLES_FILENAME}",
        target,
        auto_download,
        f"Marigold V2 '{spec.label}' checkpoint",
    )
    embeds = _ensure_file(
        f"{EMBED_SUBDIR}/{spec.embed_prefix}_prompt_embeds.pt",
        target,
        auto_download,
        f"{spec.modality} prompt embeddings",
    )
    mask = _ensure_file(
        f"{EMBED_SUBDIR}/{spec.embed_prefix}_prompt_mask.pt",
        target,
        auto_download,
        f"{spec.modality} prompt mask",
    )
    return trainables, embeds, mask
