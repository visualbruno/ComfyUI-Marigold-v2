# Copyright 2026 Huawei Technologies Co., Ltd.
# Licensed under the Apache License, Version 2.0.
"""Loading and checkpoint switching for the Marigold V2 backbone.

Port of ``marigoldv2/experiments/20260316_qwen_depth/component_loader.py``: the
Qwen-Image-Edit-2509 DiT is quantized, a rank-128 LoRA adapter is injected, and
the per-modality ``trainables.safetensors`` supplies the LoRA weights plus, for
most checkpoints, a fine-tuned VAE decoder.

The heavy backbone is loaded once per ComfyUI session and shared: switching
modality only swaps the adapter and decoder weights.
"""

from __future__ import annotations

import fnmatch
import gc
import logging

import torch
import torch.nn as nn

from .backbone import TransformerSource
from .constants import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_INIT,
    LORA_RANK,
    LORA_TARGET_MODULES,
    QUANT_DEQUANTIZE_MODULES,
    QUANT_SKIP_MODULES,
    CheckpointSpec,
)

log = logging.getLogger("ComfyUI-Marigold-v2")

QUANTIZATION_LEVELS = ["4bit", "8bit", "none"]


# --------------------------------------------------------------------------- #
# 4-bit helpers (upstream component_loader.py)
# --------------------------------------------------------------------------- #
def _bnb4bit_to_bf16_linear(bnb_lin: nn.Module) -> nn.Linear:
    """Convert a bitsandbytes Linear4bit back to a standard bfloat16 nn.Linear."""
    from bitsandbytes.functional import dequantize_4bit

    device = next(bnb_lin.parameters()).device
    weight = dequantize_4bit(bnb_lin.weight.data, bnb_lin.weight.quant_state)
    linear = nn.Linear(
        in_features=bnb_lin.in_features,
        out_features=bnb_lin.out_features,
        bias=bnb_lin.bias is not None,
        device=device,
        dtype=torch.bfloat16,
    )
    with torch.no_grad():
        linear.weight.copy_(weight.to(device=device, dtype=torch.bfloat16))
        if bnb_lin.bias is not None:
            linear.bias.copy_(bnb_lin.bias.detach().to(device=device, dtype=torch.bfloat16))
    return linear


def _set_submodule(root: nn.Module, module_name: str, new_module: nn.Module) -> None:
    parts = module_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def _expand_module_patterns(model: nn.Module, patterns) -> list[str]:
    all_names = [name for name, _ in model.named_modules()]
    expanded: list[str] = []
    for pattern in patterns:
        if any(ch in pattern for ch in ("*", "?", "[")):
            expanded.extend(n for n in all_names if fnmatch.fnmatch(n, pattern))
        else:
            expanded.append(pattern)
    seen: set[str] = set()
    return [n for n in expanded if not (n in seen or seen.add(n))]


def _dequantize_modules_to_bf16(model: nn.Module, module_patterns) -> None:
    """Restore selected 4-bit linear layers to bf16, as done during training."""
    named_modules = dict(model.named_modules())
    replaced: list[str] = []
    for name in _expand_module_patterns(model, module_patterns):
        module = named_modules.get(name)
        if module is None:
            log.warning("[Marigold V2] dequantize target not found: %s", name)
            continue
        weight = getattr(module, "weight", None)
        if weight is not None and hasattr(weight, "quant_state"):
            new_module = _bnb4bit_to_bf16_linear(module)
            new_module.requires_grad_(False)
            _set_submodule(model, name, new_module)
            replaced.append(name)
    if replaced:
        log.info("[Marigold V2] dequantized to bf16: %s", ", ".join(replaced))


# --------------------------------------------------------------------------- #
# Backbone
# --------------------------------------------------------------------------- #
class MarigoldV2Backbone:
    """The frozen Qwen VAE + LoRA-adapted DiT, plus the active checkpoint."""

    def __init__(
        self,
        qwen_dir: str,
        source: TransformerSource,
        quantization: str,
        device: torch.device,
    ):
        self.qwen_dir = qwen_dir
        self.source = source
        # GGUF weights arrive already quantized, so the bitsandbytes level does
        # not apply to them.
        self.quantization = "gguf" if source.kind == "gguf" else quantization
        self.device = device
        self.dtype = torch.bfloat16
        self.current_checkpoint: str | None = None
        if self.quantization not in ("none", "gguf") and device.type != "cuda":
            raise RuntimeError(
                f"{quantization} quantization needs a CUDA device, but ComfyUI is "
                f"running on '{device}'. Set quantization to 'none' to run in bf16."
            )

        self.vae = self._load_vae()
        # The fine-tuned decoders shipped with the checkpoints overwrite these;
        # keeping a pristine copy lets Log-stage1 (decoder not trained) and
        # modality switches restore the original weights.
        self._pristine_vae = {
            key: value.detach().to("cpu", copy=True)
            for key, value in self.vae.state_dict().items()
            if key.startswith(("decoder.", "post_quant_conv."))
        }
        self.transformer = self._load_transformer()

    # -- loading ----------------------------------------------------------- #
    def _load_vae(self):
        from diffusers import AutoencoderKLQwenImage

        log.info("[Marigold V2] Loading Qwen-Image-Edit VAE from %s", self.qwen_dir)
        vae = AutoencoderKLQwenImage.from_pretrained(
            self.qwen_dir,
            subfolder="vae",
            local_files_only=True,
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        vae.requires_grad_(False)
        vae.eval()
        return vae.to(self.device, dtype=self.dtype)

    def _load_transformer(self):
        log.info(
            "[Marigold V2] Loading the Qwen-Image-Edit DiT from %s (%s)",
            self.source.path,
            self.quantization,
        )
        if self.source.kind == "gguf":
            transformer = self._load_transformer_gguf()
        else:
            transformer = self._load_transformer_diffusers()

        transformer.requires_grad_(False)
        if self.quantization == "4bit":
            _dequantize_modules_to_bf16(transformer, QUANT_DEQUANTIZE_MODULES)

        self._add_lora(transformer)
        transformer.requires_grad_(False)
        transformer.eval()
        return transformer

    def _load_transformer_diffusers(self):
        from diffusers import QwenImageTransformer2DModel

        kwargs = dict(
            pretrained_model_name_or_path=self.source.path,
            subfolder="transformer",
            local_files_only=True,
            torch_dtype=self.dtype,
        )
        quantization_config = self._quantization_config()
        if quantization_config is not None:
            kwargs["quantization_config"] = quantization_config
            # bitsandbytes has to build the quantized weights on the target GPU.
            kwargs["device_map"] = self.device
            log.info(
                "[Marigold V2] Quantizing 20B parameters to %s; the first load "
                "takes a few minutes",
                self.quantization,
            )

        transformer = QwenImageTransformer2DModel.from_pretrained(**kwargs)

        if quantization_config is None:
            return transformer.to(self.device, dtype=self.dtype)

        from peft import prepare_model_for_kbit_training

        # Freezes everything and puts the unquantized layers in fp32, exactly
        # as the reference does before injecting the adapter.
        return prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)

    def _load_transformer_gguf(self):
        from diffusers import GGUFQuantizationConfig, QwenImageTransformer2DModel

        transformer = QwenImageTransformer2DModel.from_single_file(
            self.source.path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=self.dtype),
            config=self.qwen_dir,
            subfolder="transformer",
            torch_dtype=self.dtype,
            local_files_only=True,
        )
        # Everything stays bfloat16 here. The bitsandbytes path puts the
        # unquantized layers in fp32 (peft's prepare_model_for_kbit_training),
        # but the GGUF linear does not cast its inputs, so an fp32 norm feeding
        # a bf16 dequantized weight would fail outright.
        return transformer.to(self.device)

    def _quantization_config(self):
        if self.quantization == "none":
            return None
        from diffusers import BitsAndBytesConfig

        if self.quantization == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                llm_int8_skip_modules=QUANT_SKIP_MODULES,
            )
        if self.quantization == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=QUANT_SKIP_MODULES,
            )
        raise ValueError(
            f"Unsupported quantization '{self.quantization}'; "
            f"expected one of {QUANTIZATION_LEVELS}"
        )

    @staticmethod
    def _add_lora(transformer) -> None:
        from peft import LoraConfig

        # proj_out is dequantized instead of adapted; see component_loader.py.
        targets = [t for t in LORA_TARGET_MODULES if "proj_out" not in t]
        transformer.add_adapter(
            LoraConfig(
                r=LORA_RANK,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                init_lora_weights=LORA_INIT,
                target_modules=targets,
            )
        )
        for name, param in transformer.named_parameters():
            if "lora_" in name:
                param.data = param.data.to(torch.bfloat16)

    # -- checkpoints ------------------------------------------------------- #
    def apply_checkpoint(self, spec: CheckpointSpec, trainables_path: str) -> None:
        """Load one ``trainables.safetensors`` into the shared backbone."""
        if self.current_checkpoint == trainables_path:
            return

        from safetensors.torch import load_file

        log.info("[Marigold V2] Applying %s checkpoint: %s", spec.label, trainables_path)
        self.current_checkpoint = None
        state = load_file(trainables_path, device="cpu")

        transformer_state = {
            key[len("Diffuser."):]: value
            for key, value in state.items()
            if key.startswith("Diffuser.")
        }
        vae_state = {
            key[len("VAE."):]: value
            for key, value in state.items()
            if key.startswith("VAE.")
        }
        if not transformer_state:
            raise RuntimeError(
                f"{trainables_path} contains no 'Diffuser.*' weights; "
                "it is not a Marigold V2 checkpoint."
            )

        _, unexpected = self.transformer.load_state_dict(transformer_state, strict=False)
        if unexpected:
            raise RuntimeError(
                f"{len(unexpected)} checkpoint tensors do not fit the LoRA adapter "
                f"(first: {unexpected[0]}). Check the diffusers and peft versions."
            )
        log.info("[Marigold V2] loaded %d DiT tensors", len(transformer_state))

        # Restore the pristine decoder first so a checkpoint without VAE weights
        # (Log-stage1) never inherits the previous modality's decoder.
        self.vae.load_state_dict(self._pristine_vae, strict=False)
        if vae_state:
            _, unexpected = self.vae.load_state_dict(vae_state, strict=False)
            if unexpected:
                raise RuntimeError(
                    f"{len(unexpected)} VAE tensors in the checkpoint are unexpected "
                    f"(first: {unexpected[0]})."
                )
            log.info("[Marigold V2] loaded %d VAE decoder tensors", len(vae_state))
        self.vae.to(self.device, dtype=self.dtype)
        self.current_checkpoint = trainables_path


# --------------------------------------------------------------------------- #
# Session-wide cache
# --------------------------------------------------------------------------- #
_BACKBONE: MarigoldV2Backbone | None = None


def get_backbone(
    qwen_dir: str,
    source: TransformerSource,
    quantization: str,
    device: torch.device,
) -> MarigoldV2Backbone:
    global _BACKBONE
    effective = "gguf" if source.kind == "gguf" else quantization
    if (
        _BACKBONE is not None
        and _BACKBONE.qwen_dir == qwen_dir
        and _BACKBONE.source == source
        and _BACKBONE.quantization == effective
        and _BACKBONE.device == device
    ):
        return _BACKBONE
    unload_backbone()
    _BACKBONE = MarigoldV2Backbone(qwen_dir, source, quantization, device)
    return _BACKBONE


def unload_backbone() -> None:
    global _BACKBONE
    if _BACKBONE is None:
        return
    log.info("[Marigold V2] Unloading backbone")
    _BACKBONE.transformer = None
    _BACKBONE.vae = None
    _BACKBONE = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
