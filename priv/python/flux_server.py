"""
FLUX model inference module for Margarine via Pythonx (zero-copy shared memory).

Provides text-to-image generation using FLUX.1 models (schnell and dev) with
efficient tensor transfer via Pythonx's zero-copy shared memory between
Nx (Elixir) and NumPy (Python).

Architecture:
- Nx tensors → Pythonx → NumPy arrays (zero-copy, direct memory sharing)
- Returns numpy arrays as (bytes, shape, dtype) tuples for efficient transfer
- No JSON encoding overhead

Supported models:
- FLUX.1-schnell: Fast (4 steps), guidance_scale: 0.0
- FLUX.1-dev: High quality (28 steps), guidance_scale: 3.5
"""

import torch
import numpy as np
from diffusers import FluxPipeline

# Global model instances (loaded once at startup)
_models = None


def initialize_model(model_type, model_id, device="mps", torch_dtype="bfloat16", token=None):
    """
    Load FLUX models into memory once.

    Args:
        model_type: Model type string ("flux_schnell" or "flux_dev")
        model_id: HuggingFace model ID
            - "black-forest-labs/FLUX.1-schnell" (fast, 4 steps)
            - "black-forest-labs/FLUX.1-dev" (quality, 28 steps)
        device: Device to run on ("mps", "cuda", "cpu")
        torch_dtype: Data type ("bfloat16" recommended, "float16", "float32")
        token: HuggingFace API token (for gated models)

    Returns:
        dict: Status information
    """
    global _models
    import os

    # Validate inputs
    if model_type not in ["flux_schnell", "flux_dev"]:
        return {"error": f"Invalid model_type: {model_type}. Must be flux_schnell or flux_dev"}

    if device not in ["mps", "cuda", "cpu"]:
        return {"error": f"Invalid device: {device}. Must be mps, cuda, or cpu"}

    # Map string dtype to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }

    if torch_dtype not in dtype_map:
        return {"error": f"Invalid torch_dtype: {torch_dtype}. Must be bfloat16, float16, or float32"}

    dtype = dtype_map[torch_dtype]

    # Get HF token from parameter or environment
    hf_token = token or os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')

    print(f"[Margarine.FluxServer] Loading FLUX model: {model_id}")
    print(f"[Margarine.FluxServer] Device: {device}, dtype: {dtype}")
    if hf_token:
        print(f"[Margarine.FluxServer] Using HuggingFace token (length: {len(hf_token)})")
    else:
        print(f"[Margarine.FluxServer] No HuggingFace token - model must be public")
    print(f"[Margarine.FluxServer] First run will download ~30GB (cached after that)...")

    try:
        # Load full pipeline
        pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            token=hf_token
        )

        print(f"[Margarine.FluxServer] ✓ Model loaded successfully")

        # Enable CPU offload for low VRAM systems
        if device == "mps" or (device == "cuda" and torch.cuda.get_device_properties(0).total_memory < 20e9):
            print("[Margarine.FluxServer] Enabling CPU offload for VRAM optimization...")
            pipe.enable_model_cpu_offload()
        else:
            pipe = pipe.to(device)

        # Set models to eval mode
        pipe.vae.eval()
        pipe.transformer.eval()
        pipe.text_encoder.eval()
        pipe.text_encoder_2.eval()

        _models = {
            "pipe": pipe,
            "device": device,
            "dtype": dtype,
            "model_id": model_id,
            "model_type": model_type
        }

        print("[Margarine.FluxServer] FLUX model ready")

        return {
            "status": "initialized",
            "device": device,
            "dtype": str(dtype),
            "model_id": model_id,
            "model_type": model_type,
            "parameters": "12B"
        }

    except Exception as e:
        return {"error": f"Failed to load model: {str(e)}"}


def encode_prompt(prompt, negative_prompt="", guidance_scale=3.5):
    """
    Encode text prompts using CLIP + T5.

    Args:
        prompt: Positive prompt text
        negative_prompt: Negative prompt (FLUX doesn't use this typically)
        guidance_scale: CFG guidance scale (3-4 typical for FLUX.1-dev, 0.0 for schnell)

    Returns:
        dict: Encoded embeddings as (bytes, shape, dtype) tuples
    """
    if _models is None:
        return {"error": "Models not initialized. Call initialize_model() first."}

    if not isinstance(prompt, str):
        return {"error": "Prompt must be a string"}

    if not isinstance(guidance_scale, (int, float)):
        return {"error": "guidance_scale must be a number"}

    pipe = _models["pipe"]
    device = _models["device"]

    print(f"[Margarine.FluxServer] Encoding prompt: '{prompt[:50]}...'")

    try:
        with torch.no_grad():
            prompt_embeds, pooled_prompt_embeds, _ = pipe.encode_prompt(
                prompt=prompt,
                prompt_2=None,
                device=device,
                num_images_per_prompt=1,
                prompt_embeds=None,
                pooled_prompt_embeds=None,
                max_sequence_length=512,
            )

        print(f"[Margarine.FluxServer] Prompt embeds: {prompt_embeds.shape}")
        print(f"[Margarine.FluxServer] Pooled embeds: {pooled_prompt_embeds.shape}")

        # Convert to float32 for numpy compatibility
        prompt_embeds_f32 = prompt_embeds.to(dtype=torch.float32)
        pooled_embeds_f32 = pooled_prompt_embeds.to(dtype=torch.float32)

        # Convert to numpy
        prompt_embeds_np = prompt_embeds_f32.cpu().numpy()
        pooled_embeds_np = pooled_embeds_f32.cpu().numpy()

        # Return as tuples for zero-copy transfer
        return {
            "prompt_embeds": (
                prompt_embeds_np.data.tobytes(),
                list(prompt_embeds_np.shape),
                str(prompt_embeds_np.dtype)
            ),
            "pooled_embeds": (
                pooled_embeds_np.data.tobytes(),
                list(pooled_embeds_np.shape),
                str(pooled_embeds_np.dtype)
            ),
            "guidance_scale": guidance_scale
        }

    except Exception as e:
        return {"error": f"Failed to encode prompt: {str(e)}"}


def transformer_forward(latents_np, timestep, prompt_embeds_np, pooled_embeds_np, guidance_scale=3.5):
    """
    Run one forward pass through FLUX transformer.

    Args:
        latents_np: Current latent tensor [batch, channels, h, w]
        timestep: Current timestep (0.0 to 1.0 for FLUX flow matching)
        prompt_embeds_np: Text embeddings from T5
        pooled_embeds_np: Pooled embeddings from CLIP
        guidance_scale: CFG scale (3-4 for FLUX.1-dev, 0.0 for schnell)

    Returns:
        tuple: (bytes, shape, dtype) for predicted noise/velocity
    """
    if _models is None:
        return {"error": "Models not initialized. Call initialize_model() first."}

    pipe = _models["pipe"]
    device = _models["device"]
    dtype = _models["dtype"]

    try:
        # Convert inputs to torch tensors
        latents = torch.from_numpy(latents_np).to(device=device, dtype=dtype)
        prompt_embeds = torch.from_numpy(prompt_embeds_np).to(device=device, dtype=dtype)
        pooled_embeds = torch.from_numpy(pooled_embeds_np).to(device=device, dtype=dtype)

        # Pack latents into 2x2 patches (FLUX requirement)
        latents_packed = _pack_latents(latents)

        # Prepare positional encoding
        latent_image_ids = _prepare_latent_image_ids(
            batch_size=latents.shape[0],
            height=latents.shape[2],
            width=latents.shape[3],
            device=device,
            dtype=dtype
        )

        text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)

        # Timesteps in range [0, 1]
        t = torch.tensor([timestep] * latents.shape[0], device=device, dtype=torch.float32)

        # Build transformer kwargs
        transformer_kwargs = {
            "hidden_states": latents_packed,
            "timestep": t,
            "encoder_hidden_states": prompt_embeds,
            "pooled_projections": pooled_embeds,
            "img_ids": latent_image_ids,
            "txt_ids": text_ids,
            "return_dict": False
        }

        # Add guidance for FLUX.1-dev
        model_type = _models.get("model_type", "").lower()
        if "dev" in model_type:
            guidance_tensor = torch.tensor([guidance_scale] * latents.shape[0], device=device, dtype=torch.float32)
            transformer_kwargs["guidance"] = guidance_tensor

        # Forward through transformer
        with torch.no_grad():
            model_output = pipe.transformer(**transformer_kwargs)[0]

        # Unpack back to spatial format
        model_output_unpacked = _unpack_latents(
            model_output,
            height=latents.shape[2],
            width=latents.shape[3]
        )

        # Convert to float32 for numpy
        model_output_f32 = model_output_unpacked.to(dtype=torch.float32)
        model_output_np = model_output_f32.cpu().numpy()

        # Check for NaN/Inf
        nan_count = np.isnan(model_output_np).sum()
        inf_count = np.isinf(model_output_np).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"[Margarine.FluxServer] WARNING: {nan_count} NaN, {inf_count} Inf values detected")
            model_output_np = np.nan_to_num(model_output_np, nan=0.0, posinf=0.0, neginf=0.0)

        return (
            model_output_np.data.tobytes(),
            list(model_output_np.shape),
            str(model_output_np.dtype)
        )

    except Exception as e:
        return {"error": f"Transformer forward failed: {str(e)}"}


def vae_decode(latents_np):
    """
    Decode latents to pixel space using VAE decoder.

    Args:
        latents_np: Latent tensor [batch, channels, h, w]

    Returns:
        tuple: (bytes, shape, dtype) for decoded image [batch, 3, height, width]
    """
    if _models is None:
        return {"error": "Models not initialized. Call initialize_model() first."}

    pipe = _models["pipe"]
    vae = pipe.vae
    device = _models["device"]

    try:
        # Check for NaN/Inf in input
        nan_count = np.isnan(latents_np).sum()
        inf_count = np.isinf(latents_np).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"[Margarine.FluxServer] WARNING: Input has {nan_count} NaN, {inf_count} Inf")
            latents_np = np.nan_to_num(latents_np, nan=0.0, posinf=0.0, neginf=0.0)

        # Get VAE dtype
        vae_dtype = next(vae.parameters()).dtype

        # Convert to torch tensor
        latents = torch.from_numpy(latents_np).to(device=device, dtype=vae_dtype)

        # FLUX VAE scaling
        latents = latents / vae.config.scaling_factor

        # Decode
        with torch.no_grad():
            image = vae.decode(latents, return_dict=False)[0]

        # Convert to float32 for numpy
        image_f32 = image.to(dtype=torch.float32)
        image_np = image_f32.cpu().numpy()

        # Check output
        nan_count = np.isnan(image_np).sum()
        inf_count = np.isinf(image_np).sum()
        if nan_count > 0 or inf_count > 0:
            print(f"[Margarine.FluxServer] WARNING: Output has {nan_count} NaN, {inf_count} Inf")
            image_np = np.nan_to_num(image_np, nan=0.0, posinf=0.0, neginf=0.0)

        print(f"[Margarine.FluxServer] VAE decode: min={image_np.min():.4f}, max={image_np.max():.4f}")

        return (
            image_np.data.tobytes(),
            list(image_np.shape),
            str(image_np.dtype)
        )

    except Exception as e:
        return {"error": f"VAE decode failed: {str(e)}"}


def vae_encode(image_np):
    """
    Encode image to latent space using VAE encoder.

    Args:
        image_np: Image tensor [batch, 3, height, width] in range [-1, 1]

    Returns:
        tuple: (bytes, shape, dtype) for encoded latents
    """
    if _models is None:
        return {"error": "Models not initialized. Call initialize_model() first."}

    pipe = _models["pipe"]
    vae = pipe.vae
    device = _models["device"]

    try:
        vae_dtype = next(vae.parameters()).dtype
        image = torch.from_numpy(image_np).to(device=device, dtype=vae_dtype)

        with torch.no_grad():
            latents = vae.encode(image).latent_dist.sample()

        latents = latents * vae.config.scaling_factor

        # Convert to float32 for numpy
        latents_f32 = latents.to(dtype=torch.float32)
        latents_np = latents_f32.cpu().numpy()

        return (
            latents_np.data.tobytes(),
            list(latents_np.shape),
            str(latents_np.dtype)
        )

    except Exception as e:
        return {"error": f"VAE encode failed: {str(e)}"}


def generate_latents(height, width, seed=None):
    """
    Generate initial random latents for diffusion.

    Args:
        height: Image height in pixels
        width: Image width in pixels
        seed: Random seed (int, optional)

    Returns:
        tuple: (bytes, shape, dtype) for random latents [1, 16, h//8, w//8]
    """
    if _models is None:
        return {"error": "Models not initialized. Call initialize_model() first."}

    if not isinstance(height, int) or height <= 0:
        return {"error": "height must be a positive integer"}

    if not isinstance(width, int) or width <= 0:
        return {"error": "width must be a positive integer"}

    if seed is not None and (not isinstance(seed, int) or seed < 0):
        return {"error": "seed must be a non-negative integer or None"}

    device = _models["device"]
    dtype = _models["dtype"]

    # Calculate latent dimensions (FLUX uses 8x downsampling)
    latent_h = height // 8
    latent_w = width // 8
    latent_channels = 16

    print(f"[Margarine.FluxServer] Generating latents: {height}x{width}, seed={seed}")

    try:
        # Create generator with seed if provided
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))

        # Generate random latents
        latents = torch.randn(
            (1, latent_channels, latent_h, latent_w),
            generator=generator,
            device="cpu",  # Generate on CPU for seed consistency
            dtype=torch.float32
        ).to(device=device, dtype=dtype)

        # Convert to float32 for numpy
        latents_f32 = latents.to(dtype=torch.float32)
        latents_np = latents_f32.cpu().numpy()

        print(f"[Margarine.FluxServer] Latents: min={latents_np.min():.4f}, max={latents_np.max():.4f}")

        return (
            latents_np.data.tobytes(),
            list(latents_np.shape),
            str(latents_np.dtype)
        )

    except Exception as e:
        return {"error": f"Failed to generate latents: {str(e)}"}


def get_model_info():
    """
    Get current model information.

    Returns:
        dict: Model information
    """
    if _models is None:
        return {"status": "not_initialized"}

    return {
        "status": "ready",
        "device": _models["device"],
        "dtype": str(_models["dtype"]),
        "model_id": _models["model_id"],
        "model_type": _models["model_type"],
        "parameters": "12B"
    }


def clear_memory():
    """
    Force memory cleanup to prevent OOM.

    Returns:
        dict: Status information
    """
    global _models
    import gc

    print("[Margarine.FluxServer] Clearing memory...")

    if _models is not None:
        pipe = _models["pipe"]
        device = _models["device"]

        # Move models to CPU
        if device in ["cuda", "mps"]:
            print(f"[Margarine.FluxServer] Moving models from {device} to CPU...")
            pipe.to("cpu")

        # Empty device cache
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif device == "mps" and torch.backends.mps.is_available():
            print("[Margarine.FluxServer] Moved from MPS to CPU")

    # Garbage collection
    gc.collect()
    print("[Margarine.FluxServer] Memory cleanup complete")

    return {"status": "memory_cleared"}


# Helper functions for FLUX latent packing/unpacking

def _pack_latents(latents):
    """Pack latents into 2x2 patches for FLUX transformer."""
    batch_size, num_channels, height, width = latents.shape
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)
    return latents


def _unpack_latents(latents, height, width):
    """Unpack latents from sequence format back to spatial format."""
    batch_size, seq_len, packed_channels = latents.shape
    num_channels = packed_channels // 4
    latents = latents.reshape(batch_size, height // 2, width // 2, num_channels, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, num_channels, height, width)
    return latents


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    """Prepare latent image IDs for FLUX transformer positional encoding."""
    latent_image_ids = torch.zeros(height // 2, width // 2, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2, device=device, dtype=dtype)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2, device=device, dtype=dtype)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids
