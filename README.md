# ComfyUI-Marigold-v2

[Marigold V2](https://github.com/huawei-bayerlab/marigold-v2) in ComfyUI: **depth**,
**surface normals**, and **albedo** from a single image, in one diffusion step.

Marigold V2 (ACM TOG 2026 / SIGGRAPH Asia 2026) repurposes the Qwen-Image-Edit-2509
diffusion transformer into a single-step dense predictor. The frozen 20B DiT is
quantized to 4 bit and steered by a rank-128 LoRA adapter; each modality is one
~1.8 GB `trainables.safetensors`.

## Two ways to run it

**A. Use models you already have** (`example_workflows/marigold_v2_comfyui_native.json`).
Marigold V2 is a LoRA over Qwen-Image-Edit, so any Qwen-Image-Edit model you have
loaded works as the backbone. Nothing large is downloaded and ComfyUI keeps managing
memory:

```
Unet Loader (GGUF) ─┐
                    ├─> Load MarigoldV2 LoRA ─> Marigold V2 Predict ─> Preview
Load VAE ───────────┘                      └────────> (vae)
```

**B. Self-contained** (`example_workflows/marigold_v2_depth.json`). One loader node
fetches the backbone and checkpoint itself and runs them through diffusers, matching
the reference implementation exactly. Simpler graph, larger download.

The two agree closely — see [Notes on fidelity](#notes-on-fidelity).

## Nodes

| Node | What it does |
|---|---|
| **Load MarigoldV2 LoRA** | Applies a checkpoint to a ComfyUI `MODEL` (path A). Also patches the `VAE` with the checkpoint's fine-tuned decoder. |
| **Marigold V2 Model Loader** | Downloads and loads a self-contained backbone plus one checkpoint (path B). |
| **Marigold V2 Predict** | Runs the prediction from either loader. Outputs a ready-to-use `IMAGE` and the float `MARIGOLDV2_RAW`. |
| **Marigold V2 Colorize Depth** | Renders raw depth with a matplotlib colormap (`Spectral_r` reproduces the published look) plus a normalized grayscale map. |
| **Marigold V2 Save Raw (.npy)** | Writes the float32 prediction to the ComfyUI output folder, like `scripts/infer.py` upstream. |

The `image` output is already the useful one per modality:

* **depth** - grayscale, per-image min/max normalized, `near_is_bright` by default
* **normals** - camera-space unit normals as an RGB normal map
* **albedo** - sRGB albedo

`raw` carries the model's own units: affine-invariant depth `[H, W]` (log, linear, or
disparity depending on the checkpoint) and `[3, H, W]` for normals and albedo. Depth is
only defined up to an unknown scale and shift per image, so it is always normalized
before display.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/BrunoFargnoli/ComfyUI-Marigold-v2
../../python_embeded/python.exe -m pip install -r ComfyUI-Marigold-v2/requirements.txt
```

Use your ComfyUI Python (`python_embeded/python.exe`, or your venv) — not a system
Python. Or install through ComfyUI-Manager.

`peft` and `bitsandbytes` are the two dependencies most installs are missing.
OpenCV is used when present (it matches the reference resizes exactly) and the
nodes fall back to PIL/torch when it is not, so no OpenCV is force-installed.

Then drop one of `example_workflows/*.json` onto the ComfyUI canvas.

## Using your own Qwen-Image-Edit weights

Put a Marigold V2 `trainables.safetensors` in `ComfyUI/models/loras` (rename it per
modality, e.g. `marigold-v2-depth-Log-stage2.safetensors`), then wire
**Load MarigoldV2 LoRA** between your model loader and **Marigold V2 Predict**.

`modality` must match the file: it selects the frozen prompt embedding that replaces
the text encoder. Those embeddings are a few MB and are the one thing this path still
fetches.

ComfyUI's stock **Load LoRA** cannot do this itself. Its Qwen branch already
understands the `lora_A.default.weight` convention the checkpoints use, but every key
is prefixed `Diffuser.`, so all 1446 of them are skipped with
`lora key not loaded`. The node here strips that prefix, drops the training-only
`iREPAStudentProjector.*` head, and routes the 108 `VAE.*` tensors to the VAE
instead — they are a fine-tuned decoder, not a LoRA, and ComfyUI's Qwen VAE
names those weights differently (`decoder.conv_in` vs `decoder.conv1`), so they
are renamed through a table in `marigold_v2/vae_keys.py`.

Any Qwen-Image-Edit model works, GGUF included. The checkpoints were trained against
2509; 2511 and 2512 load too (see the note below).

## Weights

With `auto_download` enabled (the default) the loader fetches what it needs on the
first run into `ComfyUI/models/marigold-v2`:

```
ComfyUI/models/marigold-v2/
├── Qwen-Image-Edit-2509/        from Qwen/Qwen-Image-Edit-2509
│   ├── vae/                     ~0.25 GB   always needed
│   ├── transformer/config.json             always needed
│   └── transformer/*.safetensors ~41 GB    only for the bf16 backbone
├── gguf/                        GGUF backbones downloaded by these nodes
└── Marigold-V2/                 from huawei-bayerlab/marigold-v2-0
    ├── depth/Log-stage2/trainables.safetensors    ~1.8 GB, one per checkpoint
    ├── normals/trainables.safetensors
    ├── albedo/trainables.safetensors
    └── qwen_text_embeddings/    precomputed prompts, a few MB
```

Downloads resume if interrupted; re-run the workflow to continue. Point
`$MARIGOLD_V2_MODELS_DIR` at another disk to move the whole tree, or add
`marigold-v2` to `extra_model_paths.yaml`.

### Choosing a backbone

The `backbone` widget picks where the frozen 20B DiT comes from. The VAE and the
DiT config (~250 MB) are downloaded either way; only the transformer weights differ.

| Backbone | Download | VRAM for weights | Notes |
|---|---|---|---|
| `Qwen-Image-Edit-2509 bf16` | 41 GB | ~13 GB at `4bit` | The reference setup. Quantized to NF4 on load, which takes a few minutes the first time. |
| `Qwen-Image-Edit-2509 GGUF Q8_0 … Q3_K_M` | 10–22 GB | file size + ~1.9 GB | Same 2509 model, different quantization. Loads in well under a minute. |
| A `.gguf` already on disk | 0 | file size + ~1.9 GB | Any Qwen GGUF in `models/diffusion_models`, `models/unet`, or `models/marigold-v2/gguf` is offered by name. |

NF4 is more compact than `Q4_K_M`, so on a 16 GB card the bf16 backbone actually
leaves *more* room for activations than a Q4 GGUF does — the GGUF's advantage is
the download, not the VRAM. `Q3_K_M` (10 GB) is the smallest sensible GGUF.

**On non-2509 GGUFs:** the adapter was trained against Qwen-Image-Edit-2509, but the
Qwen-Image-Edit DiT has not changed shape since, so a 2511 or 2512 GGUF loads and
runs. On a spot check (one image, same Q4_K_M quantization) 2511 tracked 2509 at
Pearson r = 0.998, mean absolute difference 1.5% of the depth range, with no loss of
fine detail. That is one image, not a benchmark — it is not the configuration the
paper measured, and nothing here was scored against the published numbers. The
loader logs a warning when the filename does not say 2509.

To pre-download by hand, place the `vae/` folder and `transformer/config.json` of
[Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509), a GGUF
from [QuantStack/Qwen-Image-Edit-2509-GGUF](https://huggingface.co/QuantStack/Qwen-Image-Edit-2509-GGUF)
(or the full `transformer/`), and the files of
[huawei-bayerlab/marigold-v2-0](https://huggingface.co/huawei-bayerlab/marigold-v2-0)
in the layout above, then turn `auto_download` off.

## VRAM and speed

The upstream figures are roughly **17 GB at 1024x1024** and **29 GB at 2048x2048**
with the 4-bit DiT. Weights alone are ~13 GB (bf16 backbone at `4bit`, LoRA
included) or file size + ~1.9 GB for a GGUF, and the rest is activations, which
scale with the pixel count.

If you are short on VRAM, lower `max_side`, pick a smaller GGUF, or enable
`vae_tiling` (a small quality cost at the tile seams). `quantization: none` runs the
bf16 backbone unquantized and needs about 45 GB.

The first bf16 load quantizes 20B parameters and takes a few minutes; a GGUF loads in
well under a minute. Either way the backbone then stays resident for the ComfyUI
session, and switching modality only swaps the adapter and the decoder (about a
second).

Images are processed one at a time, so a batch costs linearly but never runs out of
memory because of its size.

## Checkpoints

`modality: depth` exposes every released depth parameterization through
`depth_checkpoint`:

| Checkpoint | Output |
|---|---|
| `Log-stage2` (default) | affine-invariant log depth — the paper model |
| `Log-stage1` | the stage-1 model, before SinkLoss and decoder fine-tuning |
| `Log-layered` | see-through log depth: geometry behind glass |
| `Uniform-base` | affine-invariant linear depth (Marigold V1 style) |
| `Uniform-layered` | see-through linear depth |
| `Disparity-base` | affine-invariant inverse depth |
| `Disparity-layered` | see-through inverse depth |

Log and linear depth grow with distance, disparity shrinks; the nodes know which is
which, so `near_is_bright` means the same thing for every checkpoint.

`modality: normals` and `modality: albedo` each have a single checkpoint and ignore
`depth_checkpoint`.

## Notes on fidelity

The pipeline mirrors `scripts/infer.py` upstream: Lanczos resize to a multiple of 16,
VAE encode, one rectified-flow step at the fixed training timestep with the
precomputed prompt embeddings, VAE decode, then the per-modality output adapter.
The LoRA adapter built here matches the released checkpoints tensor for tensor
(1446 DiT tensors plus 108 VAE decoder tensors, exact shapes).

`encoder_seed` seeds the VAE encoder's posterior sample — the only stochastic step,
and its effect is small.

Two deliberate departures from the reference, both in the interest of not
surprising anyone: images are processed one at a time (upstream selects a different
prompt context per batch position, which would make a result depend on where it sat
in the batch), and predictions are resized back to the input resolution by default
even in `fixed` mode.

### The two engines, measured

Same image, same 2509 Q4_K_M weights, same checkpoint, depth aligned by least
squares (depth is affine-invariant, so scale and shift are free):

| Comparison | Pearson r | mean residual |
|---|---|---|
| Same engine, two different `encoder_seed` values | 0.99998 | 0.08% of range |
| **ComfyUI engine vs diffusers engine** | **0.99959** | **0.42% of range** |

So the engines differ by about five times the run-to-run noise, and well under half
a percent overall — the residual of two independent bf16 implementations of a 20B
transformer with different GGUF dequantization. Visually they are the same map. The
ComfyUI engine was also about twice as fast and used ~2 GB less VRAM in that test.

What has actually been checked: the adapter and decoder key sets against all released
checkpoints; end-to-end depth, normals, and albedo on real weights; modality
switching on a shared backbone; and the stage-by-stage agreement above (encode
r=0.99999, transformer r=0.99988, decode r=0.99958). Neither path has been scored
against the published benchmark numbers.

## Credits

Marigold V2 by Igor Pavlovic, Thiemo Wandel, Anton Obukhov, Luca Bartolomei,
Andrey Davydov, Fabio Tosi, Matteo Poggi, Sabine Süsstrunk, and Dengxin Dai
(EPFL, HUAWEI Bayer Lab, University of Bologna).

```bibtex
@article{marigoldv2,
  title   = {Marigold V2: Revisiting Diffusion Transformers for Monocular Depth Estimation},
  author  = {Pavlovic, Igor and Wandel, Thiemo and Obukhov, Anton and Bartolomei, Luca
             and Davydov, Andrey and Tosi, Fabio and Poggi, Matteo and
             S{\"u}sstrunk, Sabine and Dai, Dengxin},
  journal = {ACM Transactions on Graphics},
  year    = {2026},
  doi     = {10.1145/3842528}
}
```

Apache 2.0, same as upstream. See `NOTICE`. The model weights are distributed
separately by their authors under their own terms.
