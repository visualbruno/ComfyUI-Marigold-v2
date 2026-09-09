# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""Turning raw Marigold V2 predictions into ComfyUI IMAGE tensors."""

from __future__ import annotations

import numpy as np
import torch

from .imaging import linear_resize_hwc

COLORMAPS = [
    "Spectral",
    "Spectral_r",
    "turbo",
    "viridis",
    "magma",
    "inferno",
    "plasma",
    "gray",
]


def _get_cmap(name: str):
    import matplotlib

    try:  # matplotlib >= 3.5, the only API left in 3.9+
        return matplotlib.colormaps[name]
    except (AttributeError, KeyError):
        from matplotlib import cm

        return cm.get_cmap(name)


def normalize_depth(
    depth: np.ndarray, near_is_bright: bool, far_is_high: bool, percentile_clip: float = 0.0
) -> np.ndarray:
    """Min-max normalize an affine-invariant depth map into [0, 1].

    ``far_is_high`` describes the checkpoint (log and linear depth grow with
    distance, disparity shrinks); ``near_is_bright`` is what the user wants.
    """
    values = depth.astype(np.float32)
    finite = np.isfinite(values)
    if not np.any(finite):
        return np.zeros_like(values, dtype=np.float32)

    if percentile_clip > 0.0:
        low, high = np.percentile(
            values[finite], [percentile_clip, 100.0 - percentile_clip]
        )
    else:
        low, high = float(np.min(values[finite])), float(np.max(values[finite]))

    if high > low:
        normalized = (values - low) / (high - low)
    else:
        normalized = np.zeros_like(values, dtype=np.float32)
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=1.0, neginf=0.0).clip(0.0, 1.0)

    if near_is_bright == far_is_high:
        normalized = 1.0 - normalized
    return normalized.astype(np.float32)


def colorize(normalized: np.ndarray, colormap: str) -> np.ndarray:
    """Apply a matplotlib colormap to a [0, 1] map, returning HWC RGB in [0, 1]."""
    return _get_cmap(colormap)(normalized)[..., :3].astype(np.float32)


def to_image_batch(arrays: list[np.ndarray]) -> torch.Tensor:
    """Stack HWC (or HW) float arrays into a ComfyUI IMAGE tensor [B, H, W, 3].

    A ComfyUI batch must be one tensor, so predictions that came from
    differently sized inputs are resized to the first one.
    """
    if not arrays:
        return torch.zeros((0, 1, 1, 3), dtype=torch.float32)
    height, width = arrays[0].shape[:2]
    aligned = []
    for array in arrays:
        if array.shape[:2] != (height, width):
            array = linear_resize_hwc(array, (height, width))
        if array.ndim == 2:
            array = np.repeat(array[..., None], 3, axis=2)
        aligned.append(np.ascontiguousarray(array, dtype=np.float32))
    return torch.from_numpy(np.stack(aligned, axis=0)).clamp(0.0, 1.0)


def normals_to_image(normals: np.ndarray) -> np.ndarray:
    """Unit normals [3, H, W] in [-1, 1] to an RGB normal map in [0, 1]."""
    return ((normals.transpose(1, 2, 0) + 1.0) * 0.5).clip(0.0, 1.0).astype(np.float32)


def albedo_to_image(albedo: np.ndarray) -> np.ndarray:
    """sRGB albedo [3, H, W] to HWC in [0, 1]."""
    return albedo.transpose(1, 2, 0).clip(0.0, 1.0).astype(np.float32)
