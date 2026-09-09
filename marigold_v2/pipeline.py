# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""The Marigold V2 single-step inference pass.

Port of the ``QwenImageEncode`` -> ``QwenImageEdit2509Step`` -> ``QwenImageDecode``
-> ``Folder*Prediction`` graph that ``scripts/infer.py`` assembles upstream.
Images are processed one at a time, matching the reference dataloader
(``effective_batch_size: 1``) so a result never depends on its position in a
ComfyUI batch.
"""

from __future__ import annotations

import contextlib
import functools
import inspect
import logging

import numpy as np
import torch

from .constants import TIMESTEP, CheckpointSpec
from .imaging import lanczos_resize_chw, linear_resize_chw

log = logging.getLogger("ComfyUI-Marigold-v2")

RESOLUTION_MODES = ["native", "max_side", "fixed"]
SIZE_MULTIPLE = 16


# --------------------------------------------------------------------------- #
# Prompt embeddings (replace the 7B text encoder)
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=4)
def _load_prompt(embeds_path: str, mask_path: str):
    def load(path):
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            return torch.load(path, map_location="cpu", weights_only=False)

    embeds = load(embeds_path).contiguous()  # [N_ctx, L, D]
    mask = load(mask_path).contiguous()  # [N_ctx, L]
    if mask.dtype != torch.bool:
        mask = mask > 0
    # Batch size is always 1 here, so the reference always takes context 0.
    return embeds[:1], mask[:1]


# --------------------------------------------------------------------------- #
# Resizing (upstream dataset/dataloading/transform.py)
# --------------------------------------------------------------------------- #
def _round_up(value: int, multiple: int = SIZE_MULTIPLE) -> int:
    return max(multiple, (int(value) + multiple - 1) // multiple * multiple)


def target_size(
    height: int,
    width: int,
    mode: str,
    max_side: int,
    fixed_width: int,
    fixed_height: int,
) -> tuple[int, int]:
    """Inference resolution, always a multiple of 16 as the DiT requires."""
    if mode == "fixed":
        return _round_up(fixed_height), _round_up(fixed_width)
    if mode == "max_side" and max(height, width) > max_side > 0:
        scale = max_side / float(max(height, width))
        return _round_up(round(height * scale)), _round_up(round(width * scale))
    if mode not in RESOLUTION_MODES:
        raise ValueError(f"Unknown resolution mode '{mode}'; expected {RESOLUTION_MODES}")
    return _round_up(height), _round_up(width)


# --------------------------------------------------------------------------- #
# Latent helpers
# --------------------------------------------------------------------------- #
def _latent_stats(vae, ref: torch.Tensor):
    shape = (1, vae.config.z_dim, 1, 1, 1)
    mean = torch.tensor(vae.config.latents_mean, device=ref.device, dtype=ref.dtype)
    std = torch.tensor(vae.config.latents_std, device=ref.device, dtype=ref.dtype)
    return mean.view(shape), (1.0 / std).view(shape)


def _vae_scale_factor(vae) -> int:
    temporal_downsample = vae.config.get("temperal_downsample", None)
    return 2 ** len(temporal_downsample) if temporal_downsample is not None else 8


@functools.lru_cache(maxsize=4)
def _forward_accepts(cls, name: str) -> bool:
    return name in inspect.signature(cls.forward).parameters


def _dit_step(transformer, vae, latents: torch.Tensor, prompt) -> torch.Tensor:
    """One rectified-flow step at the fixed training timestep."""
    from diffusers import QwenImageEditPipeline

    prompt_embeds_cpu, prompt_mask_cpu = prompt
    device = next(transformer.parameters()).device
    run_dtype = torch.bfloat16

    latents_4d = latents[:, :, 0]
    batch, channels, height, width = latents_4d.shape

    prompt_embeds = prompt_embeds_cpu.to(device=device, dtype=run_dtype, non_blocking=True)
    prompt_mask = prompt_mask_cpu.to(device=device, dtype=torch.bool, non_blocking=True)

    packed = QwenImageEditPipeline._pack_latents(
        latents_4d,
        batch_size=batch,
        num_channels_latents=channels,
        height=height,
        width=width,
    ).to(run_dtype)

    timestep = torch.full((batch,), TIMESTEP, device=device, dtype=run_dtype) / 1000.0
    call_kwargs = dict(
        hidden_states=packed,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        encoder_hidden_states_mask=prompt_mask,
        img_shapes=[[(1, height // 2, width // 2)]] * batch,
        guidance=None,
        attention_kwargs=getattr(transformer, "attention_kwargs", None) or {},
    )
    # diffusers dropped txt_seq_lens from the forward signature after 0.38.
    if _forward_accepts(type(transformer), "txt_seq_lens"):
        call_kwargs["txt_seq_lens"] = prompt_mask.sum(dim=1).tolist()

    cache_ctx = (
        transformer.cache_context("cond")
        if hasattr(transformer, "cache_context")
        else contextlib.nullcontext()
    )
    with cache_ctx:
        model_pred = transformer(**call_kwargs, return_dict=False)[0]

    scale = _vae_scale_factor(vae)
    model_pred = QwenImageEditPipeline._unpack_latents(
        model_pred,
        height=height * scale,
        width=width * scale,
        vae_scale_factor=scale,
    )
    # The models are trained to predict velocity at a single fixed timestep.
    return latents.to(vae.dtype) - model_pred.to(vae.dtype)


# --------------------------------------------------------------------------- #
# Modality output adapters (upstream validation/folder_steps.py)
# --------------------------------------------------------------------------- #
def _adapt(decoded: torch.Tensor, modality: str) -> torch.Tensor:
    """Turn the decoded [1, 3, H, W] image in [-1, 1] into the modality output."""
    if modality == "depth":
        return decoded.float().mean(dim=1, keepdim=True)
    if modality == "normals":
        values = decoded.float()
        magnitude = torch.linalg.vector_norm(values, dim=1, keepdim=True)
        normals = values / magnitude.clamp_min(1.0e-6)
        return torch.where(magnitude > 1.0e-6, normals, torch.zeros_like(normals))
    if modality == "albedo":
        # Linear RGB in [0, 1], then the Marigold gamma-2.2 conversion to sRGB.
        albedo = (decoded.float() + 1.0) * 0.5
        return albedo.clamp_min(0.0).pow(1.0 / 2.2)
    raise ValueError(f"Unknown modality '{modality}'")


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #
# Both engines expose the same three steps, so the driver below does not care
# whether the weights live in a self-contained diffusers backbone or in
# ComfyUI's own model management.


class DiffusersEngine:
    """The self-contained backbone loaded by MarigoldV2ModelLoader."""

    def __init__(self, backbone):
        self.backbone = backbone

    def set_vae_tiling(self, enabled: bool) -> None:
        vae = self.backbone.vae
        toggle = getattr(vae, "enable_tiling" if enabled else "disable_tiling", None)
        if toggle is not None:
            toggle()
        elif enabled:
            log.warning("[Marigold V2] This diffusers version has no VAE tiling; ignoring")

    def encode(self, pixels, generator):
        vae = self.backbone.vae
        x = pixels.to(device=vae.device, dtype=vae.dtype).unsqueeze(2)
        latents = vae.encode(x).latent_dist.sample(generator=generator)
        mean, std_inv = _latent_stats(vae, latents)
        return (latents - mean) * std_inv

    def step(self, latents, prompt):
        return _dit_step(self.backbone.transformer, self.backbone.vae, latents, prompt)

    def decode(self, latents):
        vae = self.backbone.vae
        mean, std_inv = _latent_stats(vae, latents)
        return vae.decode(latents / std_inv + mean).sample[:, :, 0].float()


class ComfyEngine:
    """A ComfyUI MODEL plus VAE, driven through comfy's model management."""

    def __init__(self, model, vae):
        self.model = model
        self.vae = vae

    def set_vae_tiling(self, enabled: bool) -> None:
        if enabled:
            log.warning(
                "[Marigold V2] vae_tiling is not available on the ComfyUI engine; "
                "lower max_side instead"
            )

    def encode(self, pixels, generator):
        from . import comfy_engine

        return comfy_engine.encode(self.vae, pixels, generator)

    def step(self, latents, prompt):
        from . import comfy_engine

        return comfy_engine.dit_step(self.model, latents, prompt)

    def decode(self, latents):
        from . import comfy_engine

        return comfy_engine.decode(self.vae, latents)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
@torch.no_grad()
def predict(
    engine,
    images: torch.Tensor,
    spec: CheckpointSpec,
    prompt_paths: tuple[str, str],
    resolution_mode: str = "max_side",
    max_side: int = 1024,
    fixed_width: int = 1024,
    fixed_height: int = 1024,
    seed: int = 2025,
    keep_input_size: bool = True,
    vae_tiling: bool = False,
    on_image=None,
) -> list[np.ndarray]:
    """Run every image of a ComfyUI IMAGE batch through ``engine``.

    ``images`` is [B, H, W, 3] in [0, 1]. Returns one float32 array per image:
    [H, W] for depth, [3, H, W] for normals and albedo.
    """
    prompt = _load_prompt(*prompt_paths)
    engine.set_vae_tiling(vae_tiling)

    results: list[np.ndarray] = []
    for index in range(images.shape[0]):
        source = images[index].permute(2, 0, 1).float().cpu()  # [3, H, W] in [0, 1]
        orig_h, orig_w = int(source.shape[1]), int(source.shape[2])
        rgb_norm = source * 2.0 - 1.0

        out_h, out_w = target_size(
            orig_h, orig_w, resolution_mode, max_side, fixed_width, fixed_height
        )
        if (out_h, out_w) != (orig_h, orig_w):
            rgb_norm = lanczos_resize_chw(rgb_norm, (out_h, out_w))

        pixels = rgb_norm.unsqueeze(0)
        generator = torch.Generator(device="cpu").manual_seed((int(seed) + index) % (2**63))

        latents = engine.encode(pixels, generator)
        latents = engine.step(latents, prompt)
        decoded = engine.decode(latents)

        prediction = _adapt(decoded, spec.modality)[0].cpu().numpy().astype(np.float32)
        if spec.modality == "depth":
            prediction = prediction[0]  # [H, W]
        if keep_input_size:
            prediction = linear_resize_chw(prediction, (orig_h, orig_w))
        results.append(np.ascontiguousarray(prediction))

        del pixels, latents, decoded
        if on_image is not None:
            on_image(index + 1, images.shape[0])

    return results
