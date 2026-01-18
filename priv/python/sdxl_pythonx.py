"""
SDXL model inference module for Pythonx (zero-copy shared memory).

Provides SDXL image generation with dual CLIP encoders, UNet architecture,
and time_ids conditioning. Returns numpy arrays directly for zero-copy
shared memory integration with Nx (Elixir).

Key SDXL features:
- Dual CLIP text encoders (CLIP-ViT-L + CLIP-ViT-G)
- Pooled embeddings for conditioning
- Time IDs for resolution/crop conditioning
- Support for both text2img and img2img
- Shared VAE with FLUX (same latent space)
"""

# Suppress known benign warning from Python's multiprocessing resource_tracker
# This warning appears when Python processes are terminated by external processes (Elixir)
# See: https://github.com/apple/ml-stable-diffusion/issues/8
import warnings
warnings.filterwarnings('ignore', '.*resource_tracker.*', UserWarning)

# Import memory limiting FIRST to cap memory usage before loading models
# This prevents OOM crashes on machines with large amounts of RAM
from memory_limit import set_memory_limit_gb
set_memory_limit_gb(16)  # Cap Python process at 16GB

import torch
import numpy as np
from diffusers import StableDiffusionXLPipeline, DiffusionPipeline

# Global model instances (loaded once at startup)
_models = None


def initialize_model(model_type, model_id, device="mps", torch_dtype="float16", token=None):
    """
    Load SDXL models into memory once.

    Args:
        model_type: Model type string (e.g. "sdxl_base", "sdxl_turbo")
        model_id: HuggingFace model ID
            - "stabilityai/stable-diffusion-xl-base-1.0" (base model, 20 steps)
            - "stabilityai/sdxl-turbo" (fast model, 1 step)
        device: Device to run on ("mps", "cuda", "cpu")
        torch_dtype: Data type ("float16" recommended for SDXL, "float32")
        token: HuggingFace API token (for gated models)

    Returns:
        dict: Status information
    """
    global _models
    import os

    # Map string dtype to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16
    }
    dtype = dtype_map.get(torch_dtype, torch.float16)

    # Get HF token from parameter or environment
    hf_token = token or os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')

    print(f"[SdxlPythonx] Loading SDXL models from {model_id}...")
    print(f"[SdxlPythonx] Device: {device}, dtype: {dtype}")
    if hf_token:
        print(f"[SdxlPythonx] Using HuggingFace token (length: {len(hf_token)})")
    else:
        print(f"[SdxlPythonx] No HuggingFace token - model must be public")
    print(f"[SdxlPythonx] This will download ~7GB on first run (cached after that)...")
    print(f"[SdxlPythonx] Downloading UNet (~5GB)...")
    print(f"[SdxlPythonx] Downloading VAE (~335MB)...")
    print(f"[SdxlPythonx] Downloading text encoders (dual CLIP, ~1.5GB)...")
    print(f"[SdxlPythonx] This may take 2-5 minutes depending on connection speed...")

    # Load SDXL pipeline
    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=hf_token,
        use_safetensors=True,
        variant="fp16" if dtype == torch.float16 else None
    )

    print(f"[SdxlPythonx] ✓ All components downloaded/loaded from cache")

    # For low VRAM, enable CPU offloading
    if device == "mps" or (device == "cuda" and torch.cuda.get_device_properties(0).total_memory < 12e9):
        print("[SdxlPythonx] Enabling CPU offload for VRAM optimization...")
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    # Set models to eval mode
    pipe.vae.eval()
    pipe.unet.eval()
    pipe.text_encoder.eval()
    pipe.text_encoder_2.eval()

    # Force VAE to float32 for numerical stability on MPS
    # Float16 on MPS can produce NaN values during VAE decode
    if device == "mps" and dtype == torch.float16:
        print("[SdxlPythonx] Converting VAE to float32 for MPS numerical stability...")
        pipe.vae = pipe.vae.to(torch.float32)
        print("[SdxlPythonx] ✓ VAE set to float32")

    _models = {
        "pipe": pipe,
        "vae": pipe.vae,
        "unet": pipe.unet,
        "text_encoder": pipe.text_encoder,      # CLIP-ViT-L
        "text_encoder_2": pipe.text_encoder_2,  # CLIP-ViT-G
        "tokenizer": pipe.tokenizer,
        "tokenizer_2": pipe.tokenizer_2,
        "scheduler": pipe.scheduler,
        "device": device,
        "dtype": dtype,
        "model_type": model_type
    }

    print(f"[SdxlPythonx] ✓ SDXL models initialized successfully!")

    return {
        "status": "ready",
        "model_type": model_type,
        "model_id": model_id,
        "device": device,
        "dtype": str(dtype)
    }


def encode_prompt(prompt, negative_prompt="", guidance_scale=7.5):
    """
    Encode text prompt using dual CLIP encoders.

    SDXL uses two CLIP text encoders:
    - CLIP-ViT-L (768d hidden states)
    - CLIP-ViT-G (1280d hidden states + pooled output)

    Args:
        prompt: Positive prompt text
        negative_prompt: Negative prompt text (for classifier-free guidance)
        guidance_scale: Guidance strength (higher = follow prompt more closely)

    Returns:
        dict: Encoded embeddings
            - prompt_embeds: Concatenated hidden states [2, 77, 2048]
            - pooled_embeds: Pooled output from CLIP-ViT-G [2, 1280]
            - guidance_scale: float
    """
    if _models is None:
        raise RuntimeError("Models not initialized. Call initialize_model() first.")

    pipe = _models["pipe"]
    device = _models["device"]
    dtype = _models["dtype"]

    # Use pipeline's encoding method which handles dual encoders properly
    # SDXL concatenates CLIP-L (768) and CLIP-G (1280) = 2048 dimensions
    (
        prompt_embeds,
        negative_prompt_embeds,
        pooled_prompt_embeds,
        negative_pooled_prompt_embeds
    ) = pipe.encode_prompt(
        prompt=prompt,
        prompt_2=prompt,  # SDXL can use different prompts for each encoder
        device=device,
        num_images_per_prompt=1,
        do_classifier_free_guidance=(guidance_scale > 1.0),
        negative_prompt=negative_prompt,
        negative_prompt_2=negative_prompt
    )

    # Concatenate positive and negative for classifier-free guidance
    # Shape: [2, 77, 2048] where index 0 = negative, index 1 = positive
    if guidance_scale > 1.0:
        prompt_embeds_combined = torch.cat([negative_prompt_embeds, prompt_embeds])
        pooled_embeds_combined = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
    else:
        # No CFG - just use positive prompt
        prompt_embeds_combined = prompt_embeds
        pooled_embeds_combined = pooled_prompt_embeds

    # Convert to numpy arrays
    prompt_embeds_np = prompt_embeds_combined.detach().cpu().numpy()
    pooled_embeds_np = pooled_embeds_combined.detach().cpu().numpy()

    # Return as tuples for zero-copy transfer: (data_bytes, shape, dtype)
    # Pythonx will pass binary data by reference, avoiding copies
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


def get_time_ids(height, width, original_height=None, original_width=None, guidance_scale=7.5):
    """
    Get time IDs for SDXL conditioning.

    SDXL requires additional conditioning beyond text:
    - Original resolution (before resizing)
    - Crop coordinates (for training, usually 0,0 for inference)
    - Target resolution (generation size)

    Args:
        height: Target image height
        width: Target image width
        original_height: Original height (defaults to target height)
        original_width: Original width (defaults to target width)
        guidance_scale: Guidance strength (determines if we need to duplicate for CFG)

    Returns:
        numpy array: Time IDs [batch, 6] where batch=2 if CFG, else 1
    """
    if _models is None:
        raise RuntimeError("Models not initialized. Call initialize_model() first.")

    device = _models["device"]
    dtype = _models["dtype"]

    # Default to target size if not specified
    original_height = original_height or height
    original_width = original_width or width

    # SDXL time IDs: [original_height, original_width, crop_top, crop_left, target_height, target_width]
    # Crop coordinates are 0,0 for centered inference (no cropping)
    add_time_ids = torch.tensor([
        [original_height, original_width, 0, 0, height, width]
    ], dtype=dtype, device=device)

    # Duplicate for classifier-free guidance only if using CFG
    # Shape: [2, 6] if CFG, [1, 6] if no CFG
    if guidance_scale > 1.0:
        add_time_ids = torch.cat([add_time_ids, add_time_ids], dim=0)

    # Convert to numpy and return as tuple for zero-copy transfer
    time_ids_np = add_time_ids.detach().cpu().numpy()
    return (
        time_ids_np.data.tobytes(),
        list(time_ids_np.shape),
        str(time_ids_np.dtype)
    )


def unet_forward(latents_np, timestep, prompt_embeds_np, pooled_embeds_np, time_ids_np, guidance_scale):
    """
    Forward pass through SDXL UNet.

    Args:
        latents_np: Current latents [batch, 4, h, w]
        timestep: Current timestep (float or int)
        prompt_embeds_np: Text embeddings [2, 77, 2048]
        pooled_embeds_np: Pooled embeddings [2, 1280]
        time_ids_np: Time IDs [2, 6]
        guidance_scale: Guidance strength

    Returns:
        numpy array: Predicted noise [batch, 4, h, w]
    """
    if _models is None:
        raise RuntimeError("Models not initialized. Call initialize_model() first.")

    unet = _models["unet"]
    device = _models["device"]
    dtype = _models["dtype"]

    # Convert inputs to torch tensors
    # Copy if not writable (Torchx backend creates read-only arrays)
    if not latents_np.flags.writeable:
        latents_np = latents_np.copy()
    if not prompt_embeds_np.flags.writeable:
        prompt_embeds_np = prompt_embeds_np.copy()
    if not pooled_embeds_np.flags.writeable:
        pooled_embeds_np = pooled_embeds_np.copy()
    if not time_ids_np.flags.writeable:
        time_ids_np = time_ids_np.copy()

    latents = torch.from_numpy(latents_np).to(device=device, dtype=dtype)
    prompt_embeds = torch.from_numpy(prompt_embeds_np).to(device=device, dtype=dtype)
    pooled_embeds = torch.from_numpy(pooled_embeds_np).to(device=device, dtype=dtype)
    time_ids = torch.from_numpy(time_ids_np).to(device=device, dtype=dtype)

    # For classifier-free guidance, latents need to be duplicated
    # to match the batch size of prompt_embeds [2, ...]
    if guidance_scale > 1.0:
        latent_model_input = torch.cat([latents, latents])
    else:
        latent_model_input = latents

    # SDXL UNet forward pass with added embeddings
    with torch.no_grad():
        noise_pred = unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={
                "text_embeds": pooled_embeds,
                "time_ids": time_ids
            },
            return_dict=False
        )[0]

    # Perform classifier-free guidance
    if guidance_scale > 1.0:
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # Convert to numpy and return as tuple for zero-copy transfer
    noise_pred_np = noise_pred.detach().cpu().numpy()
    return (
        noise_pred_np.data.tobytes(),
        list(noise_pred_np.shape),
        str(noise_pred_np.dtype)
    )


def generate_latents(height, width, seed=None):
    """
    Generate random latent noise for initialization.

    Args:
        height: Image height in pixels
        width: Image width in pixels
        seed: Random seed for reproducibility

    Returns:
        numpy array: Random latents [1, 4, height//8, width//8]
    """
    if _models is None:
        raise RuntimeError("Models not initialized. Call initialize_model() first.")

    device = _models["device"]
    dtype = _models["dtype"]

    # SDXL VAE has 8x downsampling, 4 channels
    latent_height = height // 8
    latent_width = width // 8

    # Set seed if provided
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

    # Generate random noise
    latents = torch.randn(
        (1, 4, latent_height, latent_width),
        generator=generator,
        device=device,
        dtype=dtype
    )

    # Convert to numpy and return as tuple for zero-copy transfer
    latents_np = latents.detach().cpu().numpy()
    return (
        latents_np.data.tobytes(),
        list(latents_np.shape),
        str(latents_np.dtype)
    )


def vae_decode(latents_np):
    """
    Decode latents to pixel space using VAE decoder.

    Args:
        latents_np: Latent tensor [batch, 4, h, w]

    Returns:
        numpy array: Decoded image [batch, 3, height, width] in range [-1, 1]
    """
    if _models is None:
        raise RuntimeError("Models not initialized. Call initialize_model() first.")

    vae = _models["vae"]
    device = _models["device"]

    # Use VAE's actual dtype (may be float32 even if other models are float16)
    vae_dtype = next(vae.parameters()).dtype

    # Convert to torch tensor
    # Copy if not writable (Torchx backend creates read-only arrays)
    if not latents_np.flags.writeable:
        latents_np = latents_np.copy()
    latents = torch.from_numpy(latents_np).to(device=device, dtype=vae_dtype)

    # Debug: Check for NaN/Inf before unscaling
    if torch.isnan(latents).any() or torch.isinf(latents).any():
        print(f"[VAE DEBUG] Input latents contain NaN or Inf!")

    # Unscale latents (SDXL VAE scaling factor is 0.13025, same as FLUX)
    latents = latents / vae.config.scaling_factor

    # Debug: Check for NaN/Inf after unscaling
    if torch.isnan(latents).any() or torch.isinf(latents).any():
        print(f"[VAE DEBUG] After unscaling: NaN or Inf detected!")
        print(f"[VAE DEBUG] Scaling factor: {vae.config.scaling_factor}")
        print(f"[VAE DEBUG] Latent stats: min={latents.min()}, max={latents.max()}, mean={latents.mean()}")

    # Decode
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]

    # Debug: Check output
    if torch.isnan(image).any() or torch.isinf(image).any():
        print(f"[VAE DEBUG] VAE decode produced NaN or Inf!")

    # Convert to numpy and return as tuple for zero-copy transfer
    image_np = image.detach().cpu().numpy()
    return (
        image_np.data.tobytes(),
        list(image_np.shape),
        str(image_np.dtype)
    )


def vae_encode(image_np):
    """
    Encode image to latent space using VAE encoder.

    Args:
        image_np: Image tensor [batch, 3, height, width] in range [-1, 1]

    Returns:
        numpy array: Encoded latents [batch, 4, h//8, w//8]
    """
    if _models is None:
        raise RuntimeError("Models not initialized. Call initialize_model() first.")

    vae = _models["vae"]
    device = _models["device"]

    # Use VAE's actual dtype
    vae_dtype = next(vae.parameters()).dtype

    # Convert to torch tensor
    # Copy if not writable (Torchx backend creates read-only arrays)
    if not image_np.flags.writeable:
        image_np = image_np.copy()
    image = torch.from_numpy(image_np).to(device=device, dtype=vae_dtype)

    # Encode to latent distribution
    with torch.no_grad():
        latent_dist = vae.encode(image).latent_dist
        # Use mode() for deterministic encoding (reproducibility)
        latents = latent_dist.mode()

    # Scale latents
    latents = latents * vae.config.scaling_factor

    # Convert to numpy and return as tuple for zero-copy transfer
    latents_np = latents.detach().cpu().numpy()
    return (
        latents_np.data.tobytes(),
        list(latents_np.shape),
        str(latents_np.dtype)
    )


# Convenience function for checking initialization status
def is_initialized():
    """Check if models are loaded."""
    return _models is not None


def get_model_info():
    """Get information about loaded models."""
    if _models is None:
        return {"initialized": False}

    return {
        "initialized": True,
        "model_type": _models.get("model_type"),
        "device": _models.get("device"),
        "dtype": str(_models.get("dtype"))
    }
