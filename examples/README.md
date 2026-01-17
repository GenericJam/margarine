# Margarine Examples

Example scripts demonstrating how to use Margarine for AI image generation.

## Prerequisites

1. **Install dependencies:**
   ```bash
   mix deps.get
   ```

2. **Configure Nx backend** in `config/config.exs`:
   ```elixir
   # For Apple Silicon
   config :nx, default_backend: EMLX.Backend, default_defn_options: [compiler: EMLX]

   # OR for NVIDIA/AMD
   # config :nx, default_backend: EXLA.Backend, default_defn_options: [compiler: EXLA]
   ```

3. **First run:** The first time you run any example, Margarine will automatically download:
   - Python 3.11+ (~100MB)
   - PyTorch and dependencies (~500MB)
   - FLUX model (~12GB)

   This takes 2-5 minutes. Subsequent runs are instant!

## Examples

### FLUX Examples (Fast Artistic Generation)

#### `basic.exs` - Simple FLUX Generation

The simplest way to generate an image:

```bash
elixir examples/basic.exs
```

**What it does:**
- Generates a single image with default settings
- Uses FLUX Schnell (fast, 4 steps)
- Saves to `output_basic.png`

**Good for:**
- Learning the basics
- Quick tests
- Verifying your installation works

#### `advanced.exs` - Advanced FLUX Features

Demonstrates advanced usage:

```bash
elixir examples/advanced.exs
```

**What it does:**
- Generates multiple images with different prompts
- Uses both FLUX Schnell and FLUX Dev models
- Controls randomness with seeds
- Checks environment and available memory
- Measures generation time
- Batch processing with error handling

**Good for:**
- Production workflows
- Understanding all configuration options
- Performance testing
- Error handling patterns

### SDXL Examples (Photorealistic Quality)

#### `sdxl_basic.exs` - Simple SDXL Generation

Generate photorealistic images with SDXL:

```bash
elixir examples/sdxl_basic.exs
```

**What it does:**
- Generates photorealistic images
- Uses SDXL Turbo (1 step, fast)
- Demonstrates SDXL Base option (20 steps, higher quality)
- Saves to `output_sdxl_basic.png`

**Good for:**
- Photorealistic images
- Detailed scenes and portraits
- Understanding SDXL vs FLUX differences

#### `sdxl_img2img.exs` - SDXL Image Transformation

Transform existing images with different styles:

```bash
elixir examples/sdxl_img2img.exs
```

**What it does:**
- Generates a base image
- Applies 4 different transformations with varying strengths:
  - 0.3: Subtle style changes (watercolor)
  - 0.6: Moderate transformation (oil painting)
  - 0.8: Heavy modification (cyberpunk)
  - 1.0: Complete regeneration (new image)
- Shows effect of denoising strength parameter

**Good for:**
- Understanding img2img workflow
- Style transfer experiments
- Learning denoising strength effects

### IMG2IMG Examples (Image Transformation)

#### `img2img_test.exs` - FLUX IMG2IMG Test

Transform images using FLUX:

```bash
elixir examples/img2img_test.exs
```

**What it does:**
- Uses output from `basic.exs` as init image
- Tests 3 different strengths:
  - 0.3: Subtle changes
  - 0.7: Moderate transformation
  - 1.0: Complete regeneration
- Demonstrates img2img with FLUX model

**Good for:**
- Understanding FLUX img2img
- Comparing with SDXL img2img
- Testing different modification levels

## Model Comparison: FLUX vs SDXL

### FLUX (Artistic, Creative)
- **Style**: Artistic, creative, painterly
- **Best for**: Illustrations, artistic images, creative concepts
- **Models**:
  - `flux_schnell`: 4 steps, very fast
  - `flux_dev`: 28 steps, higher quality
- **Size**: ~12GB download
- **Typical size**: 1024x1024 or larger

### SDXL (Photorealistic)
- **Style**: Photorealistic, detailed
- **Best for**: Realistic photos, portraits, detailed scenes
- **Models**:
  - `sdxl_turbo`: 1 step, extremely fast
  - `sdxl_base`: 20 steps, highest quality
- **Size**: ~7GB download
- **Typical size**: 512x512 to 1024x1024

### When to Use Which?

**Use FLUX when:**
- Creating artistic or illustrative content
- Want creative, unique interpretations
- Need larger images (1024x1024+)
- Prefer painterly or stylized results

**Use SDXL when:**
- Need photorealistic images
- Creating portraits or realistic scenes
- Want fine details and textures
- Prefer realistic lighting and materials

## IMG2IMG Denoising Strength Guide

The `denoising_strength` parameter controls how much the output differs from the input image:

- **0.0**: No change (returns original image)
- **0.1-0.3**: Subtle style changes, keeps composition very similar
- **0.4-0.6**: Moderate transformation, changes style but preserves structure
- **0.7-0.9**: Heavy modification, only rough composition remains
- **1.0**: Complete regeneration (equivalent to text2img, ignores input)

**Tips:**
- Start with 0.5 and adjust based on results
- Lower values for style transfer (keep content)
- Higher values for creative reimagining
- 1.0 is useful for using an image as inspiration only

## Tips

### Faster Generation

**FLUX:**
- Use FLUX Schnell (4 steps) instead of Dev (28 steps)
- Reduce image size: `size: {512, 512}`

**SDXL:**
- Use SDXL Turbo (1 step) instead of Base (20 steps)
- Start with 512x512 for testing
- Use GPU acceleration (EMLX or EXLA)

### Higher Quality

**FLUX:**
- Use FLUX Dev: `model: :flux_dev`
- Increase steps: `steps: 28`
- Use full resolution: `size: {1024, 1024}` or larger

**SDXL:**
- Use SDXL Base: `model: :sdxl_base`
- Use 20+ steps: `steps: 20`
- Increase guidance scale: `guidance_scale: 7.5`
- Use full resolution: `size: {1024, 1024}`

### Reproducible Results

Always use the same seed:

```elixir
opts = [seed: 42, model: :flux_schnell]
{:ok, image} = Margarine.generate("a cat", opts)
# Always generates the same cat!
```

### Memory Management

Check available memory before generation:

```elixir
{:ok, info} = Margarine.Memory.check_available()
available_gb = info.available_bytes / 1_073_741_824

if available_gb < 16 do
  IO.puts("Warning: Less than 16GB RAM available")
  # Consider reducing image size
end
```

## Common Issues

### "Python environment not initialized"

Wait 2-5 minutes on first run for Python and dependencies to download.

### Out of Memory

- Reduce image size
- Use FLUX Schnell instead of Dev
- Close other applications
- Check available RAM: `Margarine.Memory.check_available()`

### Slow Performance

- Ensure GPU backend is configured (EMLX for Apple Silicon, EXLA for NVIDIA)
- Use FLUX Schnell (4 steps) instead of Dev (28 steps)
- First run downloads models which takes time

## Creating Your Own Examples

Copy one of the existing examples and modify:

```elixir
Mix.install([
  {:margarine, path: "."},
  {:emlx, "~> 0.1"}
])

# Your code here
prompt = "your amazing prompt"
{:ok, image} = Margarine.generate(prompt, seed: 42)
Margarine.Image.save(image, "output.png")
```

## Next Steps

- Read the [main README](../README.md)
- Check the [API documentation](https://hexdocs.pm/margarine)
- Explore the [CLAUDE.md](../CLAUDE.md) for architecture details
- Join the Elixir Forum to share your creations!
