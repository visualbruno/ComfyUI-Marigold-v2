# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""Running Marigold V2 on ComfyUI's own Qwen-Image model and VAE.

The alternative to the self-contained diffusers backbone: take a MODEL from any
Qwen-Image-Edit loader (including Unet Loader (GGUF)) and a VAE from Load VAE,
apply the Marigold adapter through ComfyUI's LoRA machinery, and let ComfyUI
decide what lives on the GPU. Nothing is downloaded except the few-MB prompt
embeddings.

ComfyUI's ``comfy.ldm.qwen_image`` uses the same module names as diffusers, and
its LoRA loader already understands the ``lora_A.default.weight`` convention the
checkpoints are saved in, so only the ``Diffuser.`` prefix has to come off. The
VAE decoder needs a real rename, which lives in ``vae_keys.py``.
"""

from __future__ import annotations

import copy
import logging

import torch

from .constants import LATENTS_MEAN, LATENTS_STD, TIMESTEP
from .vae_keys import to_comfy_vae_keys

log = logging.getLogger("ComfyUI-Marigold-v2")

DIFFUSER_PREFIX = "Diffuser."
VAE_PREFIX = "VAE."


# --------------------------------------------------------------------------- #
# Checkpoint splitting
# --------------------------------------------------------------------------- #
def split_checkpoint(state: dict) -> tuple[dict, dict]:
    """Split a ``trainables.safetensors`` into a ComfyUI LoRA and VAE weights.

    The third group in the file, ``iREPAStudentProjector.*``, is a training-only
    auxiliary head and is dropped.
    """
    lora = {
        key[len(DIFFUSER_PREFIX) :]: value
        for key, value in state.items()
        if key.startswith(DIFFUSER_PREFIX)
    }
    vae = {
        key[len(VAE_PREFIX) :]: value
        for key, value in state.items()
        if key.startswith(VAE_PREFIX)
    }
    return lora, vae


def load_checkpoint_file(path: str) -> dict:
    import comfy.utils

    return comfy.utils.load_torch_file(path, safe_load=True)


# --------------------------------------------------------------------------- #
# Model and VAE patching
# --------------------------------------------------------------------------- #
def apply_lora(model, lora: dict, strength: float):
    """Patch a ComfyUI MODEL with the Marigold adapter."""
    import comfy.sd

    if not lora:
        raise ValueError(
            "This file has no 'Diffuser.*' tensors; it is not a Marigold V2 "
            "trainables.safetensors."
        )
    patched, _ = comfy.sd.load_lora_for_models(model, None, lora, strength, 0.0)
    return patched


def apply_vae_decoder(vae, vae_state: dict):
    """Return a copy of the VAE carrying the checkpoint's fine-tuned decoder.

    The original object is left alone: ComfyUI caches it across runs and other
    branches of the graph may still be using the stock decoder.
    """
    if not vae_state:
        return vae

    renamed, unknown = to_comfy_vae_keys(vae_state)
    if unknown:
        raise RuntimeError(
            f"{len(unknown)} VAE tensors in the checkpoint have no ComfyUI "
            f"counterpart (first: {unknown[0]})."
        )

    model_keys = set(vae.first_stage_model.state_dict().keys())
    absent = [key for key in renamed if key not in model_keys]
    if absent:
        raise RuntimeError(
            f"The loaded VAE does not have {len(absent)} of the decoder tensors "
            f"this checkpoint patches (first: {absent[0]}). Is this the Qwen "
            "Image VAE?"
        )

    patched = copy.copy(vae)
    patched.patcher = vae.patcher.clone()
    patched.patcher.add_patches(
        {key: ("set", (value,)) for key, value in renamed.items()},
        strength_patch=1.0,
        strength_model=0.0,
    )
    log.info("[Marigold V2] VAE decoder patched with %d tensors", len(renamed))
    return patched


# --------------------------------------------------------------------------- #
# Latent helpers
# --------------------------------------------------------------------------- #
def _latent_stats(device, dtype):
    shape = (1, len(LATENTS_MEAN), 1, 1, 1)
    mean = torch.tensor(LATENTS_MEAN, device=device, dtype=dtype).view(shape)
    std_inv = (1.0 / torch.tensor(LATENTS_STD, device=device, dtype=dtype)).view(shape)
    return mean, std_inv


def encode(vae, pixels: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Encode an image in [-1, 1] to normalized Marigold latents.

    ``pixels`` is [B, 3, H, W]. ComfyUI's WanVAE.encode returns only the
    posterior mean, so the encoder is driven directly to recover log-variance
    and draw the sample the reference draws.
    """
    import comfy.model_management as mm

    model = vae.first_stage_model
    memory = vae.memory_used_encode(
        (pixels.shape[0], 3, 1, pixels.shape[2], pixels.shape[3]), vae.vae_dtype
    )
    mm.load_models_gpu([vae.patcher], memory_required=memory, force_full_load=vae.disable_offload)

    x = pixels.to(device=vae.device, dtype=vae.vae_dtype).unsqueeze(2)  # [B,3,1,H,W]
    with torch.no_grad():
        features = model.encoder(x[:, :, :1], feat_cache=None, feat_idx=[0])
        mean, logvar = model.conv1(features).chunk(2, dim=1)
        logvar = logvar.float().clamp(-30.0, 20.0)
        std = (0.5 * logvar).exp()
        noise = torch.randn(
            mean.shape, generator=generator, device="cpu", dtype=torch.float32
        ).to(mean.device)
        latents = mean.float() + std * noise

    stats_mean, std_inv = _latent_stats(latents.device, latents.dtype)
    return (latents - stats_mean) * std_inv


def decode(vae, latents: torch.Tensor) -> torch.Tensor:
    """Decode normalized Marigold latents to a [B, 3, H, W] image in [-1, 1].

    ``VAE.decode`` would rescale to [0, 1] and clamp, which throws away half of
    a depth map, so the inner decoder is called directly.
    """
    import comfy.model_management as mm

    memory = vae.memory_used_decode(latents.shape, vae.vae_dtype)
    mm.load_models_gpu([vae.patcher], memory_required=memory, force_full_load=vae.disable_offload)

    mean, std_inv = _latent_stats(latents.device, latents.dtype)
    z = (latents / std_inv + mean).to(device=vae.device, dtype=vae.vae_dtype)
    with torch.no_grad():
        out = vae.first_stage_model.decode(z)
    if out.ndim == 5:
        out = out[:, :, 0]
    return out.float()


# --------------------------------------------------------------------------- #
# The single transformer step
# --------------------------------------------------------------------------- #
def dit_step(model, latents: torch.Tensor, prompt) -> torch.Tensor:
    """One rectified-flow step through ComfyUI's Qwen transformer.

    ``latents`` is [B, 16, 1, h, w], normalized. ComfyUI's implementation does
    its own patchifying and rope, and its timestep embedding carries the same
    ``scale=1000``, so the fixed t = 499/1000 goes in unchanged.
    """
    import comfy.model_management as mm

    prompt_embeds_cpu, prompt_mask_cpu = prompt
    device = model.load_device
    dtype = model.model.get_dtype()

    # Activations for one image at this resolution, plus a little headroom.
    tokens = (latents.shape[-2] // 2) * (latents.shape[-1] // 2)
    memory = model.memory_required([latents.shape[0], 16, latents.shape[-2], latents.shape[-1]])
    mm.load_models_gpu([model], memory_required=memory)

    x = latents.to(device=device, dtype=dtype)
    context = prompt_embeds_cpu.to(device=device, dtype=dtype)
    # ComfyUI converts the mask to an additive one only when it is NOT floating
    # point: (m - 1) * -inf, so valid tokens contribute 0. Handing it a float
    # mask of ones instead adds +1 to every text key's attention logit, quietly
    # biasing the text stream. Integer keeps the intended path (and bool would
    # break on the subtraction).
    mask = prompt_mask_cpu.to(device=device, dtype=torch.int32)

    timestep = torch.full(
        (x.shape[0],), TIMESTEP, device=device, dtype=torch.bfloat16
    ).to(dtype) / 1000.0

    with torch.no_grad():
        prediction = model.model.diffusion_model(
            x=x,
            timestep=timestep,
            context=context,
            attention_mask=mask,
            transformer_options={},
        )
    log.debug("[Marigold V2] DiT step over %d image tokens", tokens)
    # Velocity prediction at a single fixed timestep.
    return latents - prediction.to(latents.device, latents.dtype)
