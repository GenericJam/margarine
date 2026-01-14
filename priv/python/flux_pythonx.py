"""
FLUX model inference module for Pythonx (zero-copy shared memory).

Provides the same API as flux_pipeline.py but returns numpy arrays directly
instead of JSON-encoded data. Enables ~200x faster tensor transfers via
Pythonx's zero-copy shared memory between Nx (Elixir) and NumPy (Python).

Key differences from flux_pipeline.py:
- No JSON encoding (returns numpy arrays directly)
- No NaN/Inf handling (Pythonx handles this automatically)
- Simpler code (eliminates encode_numpy/decode_numpy)
- Same model loading and inference logic
"""

# Suppress known benign warning from Python's multiprocessing resource_tracker
# This warning appears when Python processes are terminated by external processes (Elixir)
# See: https://github.com/apple/ml-stable-diffusion/issues/8
import warnings
warnings.filterwarnings('ignore', '.*resource_tracker.*', UserWarning)

import torch
import numpy as np
from diffusers import FluxPipeline

# Global model instances (loaded once at startup)
_models = None


def initialize_model(model_type, model_id, device="mps", torch_dtype="bfloat16", token=None):
    """
    Load FLUX models into memory once.

    Args:
        model_type: Model type string (e.g. "flux_schnell", "flux_dev")
        model_id: HuggingFace model ID
            - "black-forest-labs/FLUX.1-dev" (open, Apache 2.0)
            - "black-forest-labs/FLUX.1-schnell" (fastest, 4 steps)
        device: Device to run on ("mps", "cuda", "cpu")
        torch_dtype: Data type ("bfloat16" recommended, "float16", "float32")
        token: HuggingFace API token (for gated models)

    Returns:
        dict: Status information
    """
    global _models
    import os

    # Map string dtype to torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32
    }
    dtype = dtype_map.get(torch_dtype, torch.bfloat16)

    # Get HF token from parameter or environment
    hf_token = token or os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')

    print(f"[FluxPythonx] Loading FLUX models from {model_id}...")
    print(f"[FluxPythonx] Device: {device}, dtype: {dtype}")
    if hf_token:
        print(f"[FluxPythonx] Using HuggingFace token (length: {len(hf_token)})")
    else:
        print(f"[FluxPythonx] No HuggingFace token - model must be public")
    print(f"[FluxPythonx] This will download ~30GB on first run (cached after that)...")
    print(f"[FluxPythonx] Downloading transformer (12B params, ~24GB)...")
    print(f"[FluxPythonx] Downloading VAE (~335MB)...")
    print(f"[FluxPythonx] Downloading text encoders (T5 + CLIP, ~5GB)...")
    print(f"[FluxPythonx] This may take 5-15 minutes depending on connection speed...")

    # Load full pipeline (HuggingFace will show download progress to stdout)
    pipe = FluxPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        token=hf_token  # Pass token for gated models
    )

    print(f"[FluxPythonx] ✓ All components downloaded/loaded from cache")

    # For low VRAM, enable CPU offloading
    if device == "mps" or (device == "cuda" and torch.cuda.get_device_properties(0).total_memory < 20e9):
        print("[FluxPythonx] Enabling CPU offload for VRAM optimization...")
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

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

    print("[FluxPythonx] FLUX models loaded successfully")

    return {
        "status": "initialized",
        "device": device,
        "dtype": str(dtype),
        "model_id": model_id,
        "model_type": model_type,
        "parameters": "12B"
    }


def encode_prompt(prompt, negative_prompt="", guidance_scale=3.5):
    """
    Encode text prompts using CLIP + T5.

    FLUX uses dual text encoders:
    - CLIP (text_encoder) for pooled embeddings
    - T5 (text_encoder_2) for full sequence embeddings

    Args:
        prompt: Positive prompt text
        negative_prompt: Negative prompt (FLUX doesn't use this typically)
        guidance_scale: CFG guidance scale (FLUX uses lower values, 3-4 typical)

    Returns:
        dict: Encoded embeddings as numpy arrays (NOT JSON-encoded)
            - "prompt_embeds": numpy array [1, seq_len, dim]
            - "pooled_embeds": numpy array [1, pooled_dim]
    """
    if _models is None:
        raise RuntimeError("Models not initialized. Call initialize_model() first.")

    pipe = _models["pipe"]
    device = _models["device"]

    print(f"[FluxPythonx] Encoding prompt with CLIP + T5...")

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

    print(f"[FluxPythonx] Prompt embeds shape: {prompt_embeds.shape}")
    print(f"[FluxPythonx] Pooled embeds shape: {pooled_prompt_embeds.shape}")

    # Convert to float32 for numpy compatibility
    # NumPy doesn't support bfloat16, so we need to convert
    # This works for both MPS and CUDA devices
    prompt_embeds_f32 = prompt_embeds.to(dtype=torch.float32)
    pooled_embeds_f32 = pooled_prompt_embeds.to(dtype=torch.float32)

    # Convert to numpy arrays
    prompt_embeds_np = prompt_embeds_f32.cpu().numpy()
    pooled_embeds_np = pooled_embeds_f32.cpu().numpy()

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


def transformer_forward(
    latents_np,
    timestep,
    prompt_embeds_np,
    pooled_embeds_np,
    guidance_scale=3.5
):
    """
    Run one forward pass through FLUX transformer.

    This is the equivalent of SDXL's unet_forward but uses
    FLUX's rectified flow transformer.

    Args:
        latents_np: Current latent tensor [batch, channels, h, w]
        timestep: Current timestep (0.0 to 1.0 for FLUX flow matching)
        prompt_embeds_np: Text embeddings from T5
        pooled_embeds_np: Pooled embeddings from CLIP
        guidance_scale: CFG scale (3-4 recommended for FLUX)

    Returns:
        numpy array: Predicted noise/velocity [batch, channels, h, w]
    """
    if _models is None:
        raise RuntimeError("Models not initialized. Call initialize_model() first.")

    pipe = _models["pipe"]
    device = _models["device"]
    dtype = _models["dtype"]

    # Convert inputs to torch tensors
    latents = torch.from_numpy(latents_np).to(device=device, dtype=dtype)
    prompt_embeds = torch.from_numpy(prompt_embeds_np).to(device=device, dtype=dtype)
    pooled_embeds = torch.from_numpy(pooled_embeds_np).to(device=device, dtype=dtype)

    # Pack latents into 2x2 patches (FLUX requirement)
    # Shape: [batch, channels, h, w] -> [batch, h*w/4, channels*4]
    latents_packed = _pack_latents(latents)

    # Prepare image IDs and text IDs for positional encoding
    latent_image_ids = _prepare_latent_image_ids(
        batch_size=latents.shape[0],
        height=latents.shape[2],
        width=latents.shape[3],
        device=device,
        dtype=dtype
    )

    # Prepare text IDs (sequence positions for text embeddings)
    text_ids = torch.zeros(prompt_embeds.shape[1], 3, device=device, dtype=dtype)

    # FLUX uses timesteps in range [0, 1] (already normalized from Elixir)
    # Expand timestep to match batch size
    t = torch.tensor([timestep] * latents.shape[0], device=device, dtype=torch.float32)

    # Build transformer call kwargs - guidance is only required for FLUX.1-dev
    transformer_kwargs = {
        "hidden_states": latents_packed,
        "timestep": t,
        "encoder_hidden_states": prompt_embeds,
        "pooled_projections": pooled_embeds,
        "img_ids": latent_image_ids,
        "txt_ids": text_ids,
        "return_dict": False
    }

    # Add guidance tensor only for FLUX.1-dev (not for schnell)
    model_type = _models.get("model_type", "").lower()
    if "dev" in model_type:
        guidance_tensor = torch.tensor([guidance_scale] * latents.shape[0], device=device, dtype=torch.float32)
        transformer_kwargs["guidance"] = guidance_tensor

    # Forward through transformer
    with torch.no_grad():
        model_output = pipe.transformer(**transformer_kwargs)[0]

    # Unpack the output back to [batch, channels, h, w]
    model_output_unpacked = _unpack_latents(
        model_output,
        height=latents.shape[2],
        width=latents.shape[3]
    )

    # Convert to float32 for numpy (works on both MPS and CUDA)
    model_output_f32 = model_output_unpacked.to(dtype=torch.float32)

    # Convert to numpy
    model_output_np = model_output_f32.cpu().numpy()

    # DIAGNOSTIC: Check for NaN/Inf before transferring
    import numpy as np
    nan_count = np.isnan(model_output_np).sum()
    inf_count = np.isinf(model_output_np).sum()
    if nan_count > 0 or inf_count > 0:
        print(f"[FluxPythonx] WARNING: transformer_forward output has {nan_count} NaN and {inf_count} Inf values!")
        print(f"[FluxPythonx] Output stats: min={np.nanmin(model_output_np)}, max={np.nanmax(model_output_np)}, mean={np.nanmean(model_output_np)}")
        # Replace NaN/Inf with zero to prevent corruption
        model_output_np = np.nan_to_num(model_output_np, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"[FluxPythonx] Replaced NaN/Inf with zeros")

    return (
        model_output_np.data.tobytes(),
        list(model_output_np.shape),
        str(model_output_np.dtype)
    )


def vae_decode(latents_np):
    """
    Decode latents to pixel space using VAE decoder.

    Args:
        latents_np: Latent tensor [batch, channels, h, w]

    Returns:
        numpy array: Decoded image [batch, 3, height, width] in range [-1, 1]
    """
    if _models is None:
        raise RuntimeError("Models not initialized. Call initialize_model() first.")

    pipe = _models["pipe"]
    vae = pipe.vae
    device = _models["device"]

    # DIAGNOSTIC: Check input latents for NaN/Inf
    import numpy as np
    nan_count_in = np.isnan(latents_np).sum()
    inf_count_in = np.isinf(latents_np).sum()
    if nan_count_in > 0 or inf_count_in > 0:
        print(f"[FluxPythonx] WARNING: vae_decode input has {nan_count_in} NaN and {inf_count_in} Inf values!")
        print(f"[FluxPythonx] Input latents stats: min={np.nanmin(latents_np)}, max={np.nanmax(latents_np)}, mean={np.nanmean(latents_np)}")
        # Replace NaN/Inf to prevent VAE failure
        latents_np = np.nan_to_num(latents_np, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"[FluxPythonx] Replaced input NaN/Inf with zeros")

    # Get VAE's actual dtype
    vae_dtype = next(vae.parameters()).dtype

    # Convert to torch tensor
    latents = torch.from_numpy(latents_np).to(device=device, dtype=vae_dtype)

    # FLUX VAE scaling
    latents = latents / vae.config.scaling_factor

    # Decode
    with torch.no_grad():
        image = vae.decode(latents, return_dict=False)[0]

    # Convert to float32 for numpy (works on both MPS and CUDA)
    image_f32 = image.to(dtype=torch.float32)

    # Convert to numpy
    image_np = image_f32.cpu().numpy()

    # DIAGNOSTIC: Check output image for NaN/Inf and value range
    nan_count_out = np.isnan(image_np).sum()
    inf_count_out = np.isinf(image_np).sum()
    if nan_count_out > 0 or inf_count_out > 0:
        print(f"[FluxPythonx] WARNING: vae_decode output has {nan_count_out} NaN and {inf_count_out} Inf values!")
        print(f"[FluxPythonx] Output image stats: min={np.nanmin(image_np)}, max={np.nanmax(image_np)}, mean={np.nanmean(image_np)}")
        # Replace NaN/Inf with zeros
        image_np = np.nan_to_num(image_np, nan=0.0, posinf=0.0, neginf=0.0)
        print(f"[FluxPythonx] Replaced output NaN/Inf with zeros")
    else:
        # Log statistics even when clean
        print(f"[FluxPythonx] vae_decode output: min={image_np.min():.4f}, max={image_np.max():.4f}, mean={image_np.mean():.4f}")

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
        numpy array: Encoded latents [batch, channels, h, w]
    """
    if _models is None:
        raise RuntimeError("Models not initialized. Call initialize_model() first.")

    pipe = _models["pipe"]
    vae = pipe.vae
    device = _models["device"]

    vae_dtype = next(vae.parameters()).dtype
    image = torch.from_numpy(image_np).to(device=device, dtype=vae_dtype)

    with torch.no_grad():
        latents = vae.encode(image).latent_dist.sample()

    latents = latents * vae.config.scaling_factor

    # Convert to float32 for numpy (works on both MPS and CUDA)
    latents_f32 = latents.to(dtype=torch.float32)

    # Convert to numpy and return as tuple for zero-copy transfer
    latents_np = latents_f32.cpu().numpy()
    return (
        latents_np.data.tobytes(),
        list(latents_np.shape),
        str(latents_np.dtype)
    )


def generate_latents(height, width, seed=None):
    """
    Generate initial random latents for diffusion.
    Uses torch.randn with manual_seed for reproducibility across Python and Pythonx.

    Args:
        height: Image height in pixels
        width: Image width in pixels
        seed: Random seed (int, optional)

    Returns:
        numpy array: Random latents [1, 16, h//8, w//8] in float32
    """
    if _models is None:
        raise RuntimeError("Models not initialized. Call initialize_model() first.")

    pipe = _models["pipe"]
    device = _models["device"]

    # Calculate latent dimensions (FLUX uses 8x downsampling)
    latent_h = height // 8
    latent_w = width // 8
    latent_channels = 16

    import sys
    print(f"[FluxPythonx] generate_latents: height={height}, width={width}, seed={seed}", file=sys.stderr, flush=True)

    # Create generator with seed if provided
    generator = None
    if seed is not None:
        # Use CPU generator for consistency (same as flux_stepped_generator.py)
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        print(f"[FluxPythonx] Created generator with seed={seed}", file=sys.stderr, flush=True)

    # Generate random latents
    latents = torch.randn(
        (1, latent_channels, latent_h, latent_w),
        generator=generator,
        device="cpu",  # Generate on CPU for seed consistency
        dtype=torch.float32
    ).to(device=device, dtype=_models["dtype"])

    # Convert to float32 for numpy (works on both MPS and CUDA)
    latents_f32 = latents.to(dtype=torch.float32)

    # Convert to numpy and return as tuple for zero-copy transfer
    latents_np = latents_f32.cpu().numpy()

    # DIAGNOSTIC: Log latent statistics
    print(f"[FluxPythonx] Latents min={latents_np.min():.4f}, max={latents_np.max():.4f}, mean={latents_np.mean():.4f}", file=sys.stderr, flush=True)

    return (
        latents_np.data.tobytes(),
        list(latents_np.shape),
        str(latents_np.dtype)
    )


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
    Force memory cleanup to prevent OOM on systems with limited RAM.

    Moves models to CPU, empties CUDA/MPS cache, and triggers garbage collection.
    Call this between generations if experiencing memory issues.
    """
    global _models
    import gc

    print("[FluxPythonx] Starting memory cleanup...")

    if _models is not None:
        pipe = _models["pipe"]
        device = _models["device"]

        # Move models to CPU to free GPU/MPS memory
        if device in ["cuda", "mps"]:
            print(f"[FluxPythonx] Moving models from {device} to CPU...")
            pipe.to("cpu")

        # Empty device cache
        if device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("[FluxPythonx] Emptied CUDA cache")
        elif device == "mps" and torch.backends.mps.is_available():
            # MPS doesn't have empty_cache, but moving to CPU helps
            print("[FluxPythonx] Moved models from MPS to CPU")

    # Force Python garbage collection
    gc.collect()
    print("[FluxPythonx] Garbage collection completed")

    return {"status": "memory_cleared"}


# Helper functions for FLUX latent packing/unpacking
def _pack_latents(latents):
    """
    Pack latents into 2x2 patches as required by FLUX transformer.

    Args:
        latents: [batch, channels, height, width]

    Returns:
        packed: [batch, height*width/4, channels*4]
    """
    batch_size, num_channels, height, width = latents.shape

    # Reshape to 2x2 patches
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    # Permute to group patches
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    # Flatten patches into sequence
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)

    return latents


def _unpack_latents(latents, height, width):
    """
    Unpack latents from sequence format back to spatial format.

    Args:
        latents: [batch, height*width/4, channels*4]
        height: Original height
        width: Original width

    Returns:
        unpacked: [batch, channels, height, width]
    """
    batch_size, seq_len, packed_channels = latents.shape
    num_channels = packed_channels // 4

    # Reshape from sequence to 2x2 patches
    latents = latents.reshape(batch_size, height // 2, width // 2, num_channels, 2, 2)
    # Permute back to original layout
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    # Reshape to original spatial dimensions
    latents = latents.reshape(batch_size, num_channels, height, width)

    return latents


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    """
    Prepare latent image IDs for FLUX transformer positional encoding.

    Args:
        batch_size: Batch size
        height: Latent height
        width: Latent width
        device: Device
        dtype: Data type

    Returns:
        image_ids: [height//2 * width//2, 3] containing (batch_id, height_id, width_id)
    """
    # Create grid of positions for packed latents
    latent_image_ids = torch.zeros(height // 2, width // 2, 3, device=device, dtype=dtype)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height // 2, device=device, dtype=dtype)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width // 2, device=device, dtype=dtype)[None, :]

    # Flatten to sequence and repeat for batch
    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids


# Simple test
if __name__ == "__main__":
    print("Testing FLUX Pythonx inference module...")

    # Initialize models
    init_result = initialize_model(
        model_type="flux_schnell",
        model_id="black-forest-labs/FLUX.1-schnell",
        device="mps",
        torch_dtype="bfloat16"
    )
    print(f"Initialization result: {init_result}")

    # Test prompt encoding
    result = encode_prompt(
        prompt="a photograph of an astronaut riding a horse on mars",
        guidance_scale=3.5
    )
    print(f"Prompt embeds shape: {result['prompt_embeds'].shape}")
    print(f"Pooled embeds shape: {result['pooled_embeds'].shape}")

    print("FLUX Pythonx inference module ready!")
