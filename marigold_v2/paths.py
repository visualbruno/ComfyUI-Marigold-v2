# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""Where the Marigold V2 weights live inside a ComfyUI installation.

Everything goes under ``ComfyUI/models/marigold-v2`` by default. The folder is
registered with ``folder_paths`` so it can be redirected through
``extra_model_paths.yaml``; ``$MARIGOLD_V2_MODELS_DIR`` overrides it outright.
"""

from __future__ import annotations

import logging
import os

from .constants import GGUF_DIR_NAME, MARIGOLD_DIR_NAME, QWEN_DIR_NAME

log = logging.getLogger("ComfyUI-Marigold-v2")

FOLDER_NAME = "marigold-v2"
_registered = False


def _register() -> None:
    global _registered
    if _registered:
        return
    _registered = True
    try:
        import folder_paths
    except ImportError:  # running outside ComfyUI (tests, CLI)
        return
    default = os.path.join(folder_paths.models_dir, FOLDER_NAME)
    known = folder_paths.folder_names_and_paths.get(FOLDER_NAME)
    if known is None or default not in known[0]:
        try:
            folder_paths.add_model_folder_path(FOLDER_NAME, default, is_default=True)
        except TypeError:  # older ComfyUI without is_default
            folder_paths.add_model_folder_path(FOLDER_NAME, default)


def search_roots() -> list[str]:
    """Every directory that may already hold Marigold V2 weights, best first."""
    override = os.environ.get("MARIGOLD_V2_MODELS_DIR", "").strip()
    if override:
        return [os.path.abspath(os.path.expanduser(override))]

    _register()
    roots: list[str] = []
    try:
        import folder_paths

        roots.extend(folder_paths.get_folder_paths(FOLDER_NAME))
    except Exception:  # pragma: no cover - ComfyUI always provides this
        pass
    if not roots:
        roots.append(
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
        )
    # De-duplicate while keeping order.
    seen: set[str] = set()
    unique = []
    for root in roots:
        key = os.path.normcase(os.path.abspath(root))
        if key not in seen:
            seen.add(key)
            unique.append(os.path.abspath(root))
    return unique


def download_root() -> str:
    """Directory new downloads are written to."""
    root = search_roots()[0]
    os.makedirs(root, exist_ok=True)
    return root


def find_existing(*relative_parts: str) -> str | None:
    """First existing path matching ``relative_parts`` across all search roots."""
    for root in search_roots():
        candidate = os.path.join(root, *relative_parts)
        if os.path.exists(candidate):
            return candidate
    return None


def qwen_dir(create: bool = False) -> str:
    """Directory of the Qwen-Image-Edit-2509 backbone (transformer + VAE)."""
    complete = find_existing(QWEN_DIR_NAME, "transformer", "config.json")
    if complete is not None:
        return os.path.dirname(os.path.dirname(complete))
    # A partial download resumes in place rather than starting over elsewhere.
    partial = find_existing(QWEN_DIR_NAME)
    if partial is not None:
        return partial
    path = os.path.join(download_root(), QWEN_DIR_NAME)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def marigold_dir(create: bool = False) -> str:
    """Directory of the Marigold V2 checkpoints and prompt embeddings."""
    for root in search_roots():
        candidate = os.path.join(root, MARIGOLD_DIR_NAME)
        if os.path.isdir(candidate):
            return candidate
    path = os.path.join(download_root(), MARIGOLD_DIR_NAME)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def gguf_dir(create: bool = False) -> str:
    """Where GGUF backbones downloaded by these nodes are kept."""
    existing = find_existing(GGUF_DIR_NAME)
    if existing is not None:
        return existing
    path = os.path.join(download_root(), GGUF_DIR_NAME)
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def _comfy_folders(*names: str) -> list[str]:
    try:
        import folder_paths
    except ImportError:
        return []
    folders: list[str] = []
    for name in names:
        try:
            folders.extend(folder_paths.get_folder_paths(name))
        except KeyError:
            continue
    return folders


def list_local_gguf() -> dict[str, str]:
    """Basename -> full path for every Qwen GGUF the loader can offer.

    Covers this pack's own download folder plus ComfyUI's diffusion_models and
    unet folders, so a GGUF already fetched for another workflow is reusable.
    Only files whose name mentions Qwen are listed: the DiT architecture has to
    match, and offering a Flux or Hunyuan GGUF would only invite a crash.
    """
    found: dict[str, str] = {}
    roots = [gguf_dir()] + search_roots() + _comfy_folders("diffusion_models", "unet")
    for root in roots:
        if not os.path.isdir(root):
            continue
        for entry in sorted(os.listdir(root)):
            lower = entry.lower()
            if lower.endswith(".gguf") and "qwen" in lower and entry not in found:
                found[entry] = os.path.join(root, entry)
    return found
