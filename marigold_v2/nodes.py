# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""ComfyUI nodes for Marigold V2 depth, surface normals, and albedo."""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch

from . import backbone as backbone_mod
from . import download, model, pipeline, visualize
from . import comfy_engine
from .constants import (
    BF16_BACKBONE,
    DEFAULT_DEPTH_VARIANT,
    DEPTH_PARAMETERIZATIONS,
    DEPTH_VARIANTS,
    MODALITIES,
    get_spec,
    spec_for_file,
)

log = logging.getLogger("ComfyUI-Marigold-v2")

CATEGORY = "Marigold V2"


def _torch_device() -> torch.device:
    try:
        import comfy.model_management as mm

        return mm.get_torch_device()
    except Exception:  # pragma: no cover - outside ComfyUI
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _free_other_models() -> None:
    try:
        import comfy.model_management as mm

        mm.unload_all_models()
        mm.soft_empty_cache()
    except Exception:  # pragma: no cover - outside ComfyUI
        pass


def _interrupt_check() -> None:
    try:
        import comfy.model_management as mm

        mm.throw_exception_if_processing_interrupted()
    except ImportError:  # pragma: no cover - outside ComfyUI
        pass


class MarigoldV2ModelLoader:
    """Download and load the backbone, then hand a checkpoint to the sampler."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "modality": (MODALITIES, {"default": "depth"}),
                "depth_checkpoint": (
                    list(DEPTH_VARIANTS),
                    {
                        "default": DEFAULT_DEPTH_VARIANT,
                        "tooltip": "Depth parameterization; ignored for normals and albedo.",
                    },
                ),
                "backbone": (
                    backbone_mod.options(),
                    {
                        "default": BF16_BACKBONE,
                        "tooltip": "Where the frozen Qwen-Image-Edit DiT comes "
                        "from. bf16 is the exact reference setup. A 2509 GGUF is "
                        "the same model at a third of the download. A GGUF of "
                        "another release (2511+) loads but was never what the "
                        "adapter was trained against.",
                    },
                ),
                "quantization": (
                    model.QUANTIZATION_LEVELS,
                    {
                        "default": "4bit",
                        "tooltip": "Applies to the bf16 backbone only; GGUF weights "
                        "are already quantized. 4bit matches the released models "
                        "and needs ~17 GB of VRAM at 1024x1024, 'none' ~45 GB.",
                    },
                ),
                "auto_download": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Fetch missing weights from Hugging Face into "
                        "ComfyUI/models/marigold-v2 (~41 GB backbone, ~1.8 GB per checkpoint).",
                    },
                ),
            }
        }

    RETURN_TYPES = ("MARIGOLDV2_MODEL",)
    RETURN_NAMES = ("marigold_model",)
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Loads the frozen Qwen-Image-Edit-2509 backbone and one Marigold V2 "
        "checkpoint. The backbone is shared across modalities, so switching only "
        "swaps the LoRA adapter and the VAE decoder."
    )

    def load(self, modality, depth_checkpoint, backbone, quantization, auto_download):
        spec = get_spec(modality, depth_checkpoint)
        qwen_dir, source = backbone_mod.resolve(backbone, auto_download)
        caution = backbone_mod.warn_if_not_2509(source)
        if caution:
            log.warning("[Marigold V2] %s", caution)
        trainables, embeds, mask = download.ensure_checkpoint(spec, auto_download)

        _free_other_models()
        loaded = model.get_backbone(qwen_dir, source, quantization, _torch_device())
        loaded.apply_checkpoint(spec, trainables)
        return (
            {
                "engine": "diffusers",
                "spec": spec,
                "trainables": trainables,
                "prompt_paths": (embeds, mask),
                "qwen_dir": qwen_dir,
                "source": source,
                "quantization": quantization,
            },
        )


class MarigoldV2Predict:
    """Run the single-step prediction over an image batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "marigold_model": ("MARIGOLDV2_MODEL",),
                "image": ("IMAGE",),
                "resolution_mode": (
                    pipeline.RESOLUTION_MODES,
                    {
                        "default": "max_side",
                        "tooltip": "native: the input resolution, rounded up to a "
                        "multiple of 16. max_side: downscale so the longer side is "
                        "at most max_side. fixed: exactly width x height.",
                    },
                ),
                "max_side": (
                    "INT",
                    {"default": 1024, "min": 16, "max": 8192, "step": 16},
                ),
                "width": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "height": ("INT", {"default": 1024, "min": 16, "max": 8192, "step": 16}),
                "encoder_seed": (
                    "INT",
                    {
                        "default": 2025,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "tooltip": "Seeds the VAE encoder sample, the only stochastic "
                        "step. Its effect on the result is small.",
                    },
                ),
                "keep_input_size": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Resize the prediction back to the input resolution.",
                    },
                ),
                "near_is_bright": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Depth only: bright pixels are close to the camera.",
                    },
                ),
                "vae_tiling": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "Trades quality for VRAM at high resolutions."},
                ),
                "keep_model_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Diffusers backbone only; the ComfyUI engine "
                        "leaves loading and unloading to ComfyUI.",
                    },
                ),
            },
            "optional": {
                "vae": (
                    "VAE",
                    {"tooltip": "Required when the model came from Load MarigoldV2 LoRA."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MARIGOLDV2_RAW")
    RETURN_NAMES = ("image", "raw")
    FUNCTION = "predict"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Predicts depth, surface normals, or albedo in a single diffusion step. "
        "'image' is the ready-to-use visualization; 'raw' carries the float "
        "prediction for the colorize and save nodes."
    )

    def predict(
        self,
        marigold_model,
        image,
        resolution_mode,
        max_side,
        width,
        height,
        encoder_seed,
        keep_input_size,
        near_is_bright,
        vae_tiling,
        keep_model_loaded,
        vae=None,
    ):
        spec = marigold_model["spec"]
        engine = self._engine(marigold_model, vae)

        progress = None
        try:
            from comfy.utils import ProgressBar

            progress = ProgressBar(image.shape[0])
        except ImportError:  # pragma: no cover - outside ComfyUI
            pass

        def on_image(done, total):
            if progress is not None:
                progress.update_absolute(done, total)
            _interrupt_check()

        try:
            predictions = pipeline.predict(
                engine,
                image,
                spec,
                marigold_model["prompt_paths"],
                resolution_mode=resolution_mode,
                max_side=max_side,
                fixed_width=width,
                fixed_height=height,
                seed=encoder_seed,
                keep_input_size=keep_input_size,
                vae_tiling=vae_tiling,
                on_image=on_image,
            )
        finally:
            if not keep_model_loaded and marigold_model["engine"] == "diffusers":
                model.unload_backbone()

        raw = {
            "modality": spec.modality,
            "checkpoint": spec.label,
            "far_is_high": spec.far_is_high,
            "predictions": predictions,
        }
        return (_visualize(raw, near_is_bright), raw)

    @staticmethod
    def _engine(marigold_model, vae):
        """Build the engine the incoming model was loaded for."""
        if marigold_model["engine"] == "comfy":
            if vae is None:
                raise ValueError(
                    "The ComfyUI engine needs a VAE. Wire Load VAE through "
                    "'Load MarigoldV2 LoRA' into this node's vae input."
                )
            stamped = getattr(vae, "marigold_v2_decoder", None)
            if marigold_model["has_vae_decoder"] and stamped != marigold_model["lora_path"]:
                log.warning(
                    "[Marigold V2] This checkpoint ships a fine-tuned VAE decoder, "
                    "but the VAE reaching Predict does not carry it. Route Load VAE "
                    "through 'Load MarigoldV2 LoRA' for correct output."
                )
            return pipeline.ComfyEngine(marigold_model["model"], vae)

        _free_other_models()
        loaded = model.get_backbone(
            marigold_model["qwen_dir"],
            marigold_model["source"],
            marigold_model["quantization"],
            _torch_device(),
        )
        loaded.apply_checkpoint(marigold_model["spec"], marigold_model["trainables"])
        return pipeline.DiffusersEngine(loaded)


def _visualize(raw, near_is_bright: bool) -> torch.Tensor:
    modality = raw["modality"]
    if modality == "depth":
        frames = [
            visualize.normalize_depth(p, near_is_bright, raw["far_is_high"])
            for p in raw["predictions"]
        ]
    elif modality == "normals":
        frames = [visualize.normals_to_image(p) for p in raw["predictions"]]
    else:
        frames = [visualize.albedo_to_image(p) for p in raw["predictions"]]
    return visualize.to_image_batch(frames)


class MarigoldV2ColorizeDepth:
    """Render a raw depth prediction with a matplotlib colormap."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw": ("MARIGOLDV2_RAW",),
                "colormap": (
                    visualize.COLORMAPS,
                    {
                        "default": "Spectral_r",
                        "tooltip": "Spectral_r with near_is_bright on reproduces "
                        "the published Marigold look: warm near, blue far.",
                    },
                ),
                "near_is_bright": ("BOOLEAN", {"default": True}),
                "percentile_clip": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 20.0,
                        "step": 0.1,
                        "tooltip": "Clip this percentage from both ends before "
                        "normalizing, to suppress outliers.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("colored", "grayscale")
    FUNCTION = "colorize"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Normalizes affine-invariant depth per image and applies a colormap. "
        "Spectral matches the upstream visualizations."
    )

    def colorize(self, raw, colormap, near_is_bright, percentile_clip):
        if raw["modality"] != "depth":
            raise ValueError(
                f"MarigoldV2ColorizeDepth needs a depth prediction, got '{raw['modality']}'"
            )
        normalized = [
            visualize.normalize_depth(
                prediction, near_is_bright, raw["far_is_high"], percentile_clip
            )
            for prediction in raw["predictions"]
        ]
        colored = visualize.to_image_batch(
            [visualize.colorize(n, colormap) for n in normalized]
        )
        return (colored, visualize.to_image_batch(normalized))


class MarigoldV2SaveRaw:
    """Write the float predictions as .npy files, like scripts/infer.py."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "raw": ("MARIGOLDV2_RAW",),
                "filename_prefix": ("STRING", {"default": "marigold_v2/prediction"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Saves float32 .npy files to the ComfyUI output directory: [H, W] for "
        "depth, [3, H, W] for normals and albedo, in the model's own units."
    )

    def save(self, raw, filename_prefix):
        import folder_paths

        output_dir = folder_paths.get_output_directory()
        full_path, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir
        )
        results = []
        for index, prediction in enumerate(raw["predictions"]):
            name = f"{filename}_{counter + index:05}_.npy"
            np.save(os.path.join(full_path, name), prediction)
            results.append({"filename": name, "subfolder": subfolder, "type": "output"})

        sidecar = os.path.join(full_path, f"{filename}_{counter:05}_.json")
        with open(sidecar, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "modality": raw["modality"],
                    "checkpoint": raw["checkpoint"],
                    "far_is_high": raw["far_is_high"],
                    "files": [r["filename"] for r in results],
                },
                handle,
                indent=2,
            )
        log.info("[Marigold V2] saved %d prediction(s) to %s", len(results), full_path)
        return {"ui": {"text": [f"Saved {len(results)} .npy file(s) to {full_path}"]}}



class MarigoldV2LoRALoader:
    """Apply a Marigold V2 checkpoint to a ComfyUI Qwen-Image-Edit model."""

    @classmethod
    def INPUT_TYPES(cls):
        try:
            import folder_paths

            names = folder_paths.get_filename_list("loras")
        except Exception:  # pragma: no cover - outside ComfyUI
            names = []
        return {
            "required": {
                "model": (
                    "MODEL",
                    {"tooltip": "A Qwen-Image-Edit model, e.g. from Unet Loader (GGUF)."},
                ),
                "lora_name": (
                    names,
                    {
                        "tooltip": "A Marigold V2 trainables.safetensors placed in "
                        "ComfyUI/models/loras.",
                    },
                ),
                "modality": (
                    MODALITIES,
                    {
                        "default": "depth",
                        "tooltip": "Must match the checkpoint: it selects the frozen "
                        "prompt embedding that stands in for the text encoder.",
                    },
                ),
                "depth_parameterization": (
                    list(DEPTH_PARAMETERIZATIONS),
                    {
                        "default": "log_or_linear",
                        "tooltip": "Log-* and Uniform-* checkpoints grow with distance; "
                        "Disparity-* shrink. Ignored for normals and albedo.",
                    },
                ),
                "strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01},
                ),
                "auto_download": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Fetch the few-MB prompt embeddings if they are "
                        "not already in ComfyUI/models/marigold-v2.",
                    },
                ),
            },
            "optional": {
                "vae": (
                    "VAE",
                    {
                        "tooltip": "The Qwen Image VAE. Passing it here applies the "
                        "checkpoint's fine-tuned decoder to a copy of it.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MARIGOLDV2_MODEL", "MODEL", "VAE")
    RETURN_NAMES = ("marigold_model", "model", "vae")
    FUNCTION = "load"
    CATEGORY = CATEGORY
    DESCRIPTION = (
        "Applies a Marigold V2 checkpoint to a model you already loaded in "
        "ComfyUI, so nothing large is downloaded and ComfyUI keeps managing "
        "memory. The checkpoint's LoRA goes onto the DiT and its fine-tuned "
        "decoder onto a copy of the VAE."
    )

    def load(
        self,
        model,
        lora_name,
        modality,
        depth_parameterization,
        strength,
        auto_download,
        vae=None,
    ):
        import folder_paths

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        spec = spec_for_file(modality, depth_parameterization)
        embeds, mask = download.ensure_prompt_embeddings(spec, auto_download)

        state = comfy_engine.load_checkpoint_file(lora_path)
        lora, vae_state = comfy_engine.split_checkpoint(state)
        log.info(
            "[Marigold V2] %s: %d LoRA tensors, %d VAE decoder tensors",
            lora_name,
            len(lora),
            len(vae_state),
        )

        patched_model = comfy_engine.apply_lora(model, lora, strength)
        patched_vae = vae
        if vae is not None and vae_state:
            patched_vae = comfy_engine.apply_vae_decoder(vae, vae_state)
            patched_vae.marigold_v2_decoder = lora_path
        elif vae_state and vae is None:
            log.warning(
                "[Marigold V2] %s ships a fine-tuned VAE decoder. Connect your VAE "
                "to this node so it can be applied.",
                lora_name,
            )

        bundle = {
            "engine": "comfy",
            "spec": spec,
            "model": patched_model,
            "prompt_paths": (embeds, mask),
            "lora_path": lora_path,
            "has_vae_decoder": bool(vae_state),
        }
        return (bundle, patched_model, patched_vae)

NODE_CLASS_MAPPINGS = {
    "MarigoldV2ModelLoader": MarigoldV2ModelLoader,
    "MarigoldV2LoRALoader": MarigoldV2LoRALoader,
    "MarigoldV2Predict": MarigoldV2Predict,
    "MarigoldV2ColorizeDepth": MarigoldV2ColorizeDepth,
    "MarigoldV2SaveRaw": MarigoldV2SaveRaw,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MarigoldV2ModelLoader": "Marigold V2 Model Loader",
    "MarigoldV2LoRALoader": "Load MarigoldV2 LoRA",
    "MarigoldV2Predict": "Marigold V2 Predict",
    "MarigoldV2ColorizeDepth": "Marigold V2 Colorize Depth",
    "MarigoldV2SaveRaw": "Marigold V2 Save Raw (.npy)",
}
