# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""Resizing helpers.

OpenCV is what the upstream transforms use, so it is preferred when available.
ComfyUI installs vary in whether (and under which distribution) cv2 is present,
so PIL/torch fallbacks keep the nodes working without forcing a second OpenCV
into the environment.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger("ComfyUI-Marigold-v2")

try:
    import cv2
except ImportError:  # pragma: no cover - depends on the host install
    cv2 = None
    log.info(
        "[Marigold V2] OpenCV not found; using PIL/torch resizing. Install "
        "opencv-python (or opencv-contrib-python) to match the reference exactly."
    )


def lanczos_resize_hwc(array: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Lanczos resize of an HWC float32 array."""
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if array.shape[:2] == (out_h, out_w):
        return array
    if cv2 is not None:
        out = cv2.resize(array, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        return out[:, :, None] if out.ndim == 2 else out

    from PIL import Image

    channels = [
        np.asarray(
            Image.fromarray(array[..., c], mode="F").resize(
                (out_w, out_h), Image.LANCZOS
            ),
            dtype=np.float32,
        )
        for c in range(array.shape[2])
    ]
    return np.stack(channels, axis=2)


def lanczos_resize_chw(x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
    """Lanczos resize of a CHW float tensor, as the reference transform does."""
    hwc = x.detach().to("cpu", dtype=torch.float32).permute(1, 2, 0).numpy()
    out = lanczos_resize_hwc(np.ascontiguousarray(hwc), out_hw)
    return torch.from_numpy(np.ascontiguousarray(out)).permute(2, 0, 1).contiguous()


def linear_resize_hwc(array: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of an HWC or HW float32 array."""
    out_h, out_w = int(out_hw[0]), int(out_hw[1])
    if array.shape[:2] == (out_h, out_w):
        return array
    if cv2 is not None:
        return cv2.resize(
            array.astype(np.float32, copy=False),
            (out_w, out_h),
            interpolation=cv2.INTER_LINEAR,
        )
    squeeze = array.ndim == 2
    tensor = torch.from_numpy(np.ascontiguousarray(array, dtype=np.float32))
    tensor = tensor[None, None] if squeeze else tensor.permute(2, 0, 1)[None]
    tensor = F.interpolate(tensor, (out_h, out_w), mode="bilinear", align_corners=False)
    tensor = tensor[0, 0] if squeeze else tensor[0].permute(1, 2, 0)
    return tensor.numpy()


def linear_resize_chw(array: np.ndarray, out_hw: tuple[int, int]) -> np.ndarray:
    """Bilinear resize of a 2D or CHW prediction back to the input resolution."""
    if array.shape[-2:] == (int(out_hw[0]), int(out_hw[1])):
        return array
    if array.ndim == 2:
        return linear_resize_hwc(array, out_hw)
    return np.stack(
        [linear_resize_hwc(channel, out_hw) for channel in array], axis=0
    )
