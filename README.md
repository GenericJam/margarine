# Margarine 🧈

<p align="center">
  <img src="assets/logo.png" alt="Margarine Logo" width="400"/>
</p>


AI-powered image generation for Elixir using FLUX and SDXL.

Margarine brings state-of-the-art text-to-image and image-to-image generation to the Elixir ecosystem with a clean, native API. Generate beautiful images from text prompts or transform existing images with just a few lines of code.

## Features

- 🎨 **FLUX Integration** - Artistic image generation with FLUX Schnell and FLUX Dev
- 📸 **SDXL Integration** - Photorealistic images with Stable Diffusion XL (Base and Turbo)
- 🔄 **Image-to-Image** - Transform images with both FLUX and SDXL models
- ⚡ **Zero-Copy Performance** - Pythonx integration for efficient tensor transfer between Elixir and Python
- 🚀 **Zero Configuration** - Python and dependencies installed automatically on first run via UV
- 🍎 **Apple Silicon Support** - Optimized for M-series Macs with EMLX/Metal backend
- 🔥 **CUDA Support** - NVIDIA GPU acceleration via EXLA/XLA
- 💾 **Memory Safety** - Automatic checks prevent OOM crashes
- 🧪 **Production Ready** - Comprehensive tests (148+), telemetry, and error handling
- 📊 **Type Safe** - Full Dialyzer type specs on all public functions

## Quick Start

```elixir
# Add to mix.exs
def deps do
  [
    {:margarine, "~> 0.2.0"},

    # REQUIRED: Choose ONE Nx backend based on your hardware
    {:emlx, "~> 0.1"}  # For Apple Silicon (M1/M2/M3/M4)
    # {:exla, "~> 0.9"}  # For NVIDIA/AMD GPU or CPU
  ]
end
```

**Important:** You must install either EMLX or EXLA alongside Margarine. The backend handles GPU/CPU acceleration for tensor operations.

```elixir
# Simple text-to-image with FLUX
{:ok, image} = Margarine.generate("a red panda eating bamboo")
Margarine.Image.save(image, "panda.png")

# Photorealistic with SDXL
{:ok, image} = Margarine.generate("a serene mountain landscape at sunset",
  model: :sdxl_turbo,   # Fast photorealistic generation
  steps: 1
)

# Image-to-image transformation
{:ok, image} = Margarine.img2img(
  "turn into a watercolor painting",
  "photo.png",
  denoising_strength: 0.5,  # How much to change (0.0-1.0)
  model: :sdxl_base
)
```

## 🚀 Interactive Tutorials (Recommended!)

**The best way to learn Margarine is through our interactive Livebook tutorials.** These step-by-step guides run live code in your browser and let you experiment as you learn.

### Start Here: Choose Your Model

We have two comprehensive tutorials, one for each model type:

#### 📘 FLUX Tutorial (Artistic Generation)
```bash
livebook open notebooks/flux_getting_started.livemd
```

**You'll learn:**
- Generate your first artistic image with FLUX
- Understand FLUX Schnell (fast) vs Dev (quality)
- Control generation with seeds, steps, and size
- Compare variations with different parameters
- Sequential batch generation

**Best for:** Illustrations, artistic images, creative concepts

#### 📗 SDXL Tutorial (Photorealistic Generation)
```bash
livebook open notebooks/sdxl_getting_started.livemd
```

**You'll learn:**
- Generate photorealistic images with SDXL
- Use SDXL Turbo (1 step) vs Base (20 steps)
- Transform images with img2img
- Control denoising strength for style transfer
- Understand SDXL vs FLUX differences

**Best for:** Realistic photos, detailed scenes, portraits

### First Time Using Livebook?

Install it first:
```bash
mix escript.install hex livebook
```

Then open either tutorial above. The notebooks include everything you need, including dependency installation!

## Installation

### 1. Add Dependencies

Add `margarine` and an Nx backend to your `mix.exs`:

```elixir
def deps do
  [
    {:margarine, "~> 0.1.0"},

    # Choose ONE backend:
    {:emlx, "~> 0.1"}   # Apple Silicon (M1/M2/M3/M4) - Recommended for Macs
    # {:exla, "~> 0.9"}  # NVIDIA/AMD GPU or CPU
    # {:torchx, "~> 0.7"}  # PyTorch backend (experimental)
  ]
end
```

### 2. Configure Nx Backend

In `config/config.exs`:

```elixir
# For Apple Silicon
config :nx,
  default_backend: EMLX.Backend,
  default_defn_options: [compiler: EMLX]

# OR for NVIDIA/AMD GPU
# config :nx,
#   default_backend: EXLA.Backend,
#   default_defn_options: [compiler: EXLA]
```

### 3. First Run Setup

**No manual installation required!** Margarine automatically handles everything:

1. **UV Package Manager** - Automatically downloaded and installed by Pythonx
2. **Python 3.11+** - Downloaded via UV (~100MB)
3. **PyTorch & Dependencies** - Installed via UV (~500MB)
4. **Everything cached** - Instant subsequent runs

**This takes 2-5 minutes on first run.** All you need is an internet connection and ~15GB disk space.

For production deployments, we recommend running a "warm-up" generation when your server starts:

```elixir
# In your application startup
defmodule MyApp.Application do
  def start(_type, _args) do
    # Warm up Margarine to avoid first-request timeout
    Task.start(fn ->
      Margarine.check_environment()
    end)

    # ... rest of your application setup
  end
end
```

## System Requirements

### Minimum Requirements
- **Elixir**: 1.14 or later
- **RAM**: 16GB (for CPU inference)
- **Storage**: 15GB (for models and dependencies)
- **OS**: macOS (Apple Silicon recommended), Linux, or Windows

### Recommended for Best Performance
- **GPU**: Apple M-series, NVIDIA RTX 3060+ (12GB+ VRAM), or AMD equivalent
- **RAM**: 32GB
- **Storage**: SSD with 20GB+ free space

### Model Memory Requirements
- **FLUX Schnell**: ~12GB VRAM (GPU) or ~16GB RAM (CPU)
- **FLUX Dev**: ~12GB VRAM (GPU) or ~16GB RAM (CPU)
- **SDXL Turbo**: ~7GB VRAM (GPU) or ~12GB RAM (CPU)
- **SDXL Base**: ~7GB VRAM (GPU) or ~12GB RAM (CPU)

## Usage Examples

### Basic Generation

```elixir
# Simple prompt
{:ok, image} = Margarine.generate("a cute red panda")

# Check the result
IO.inspect(Nx.shape(image))  # {1024, 1024, 3}
IO.inspect(Nx.type(image))   # {:u, 8} (RGB values 0-255)

# Save to file
Margarine.Image.save(image, "panda.png")
```

### Reproducible Generation

```elixir
# Use a seed for reproducible results
opts = [seed: 42, model: :flux_schnell]

{:ok, image1} = Margarine.generate("a mountain landscape", opts)
{:ok, image2} = Margarine.generate("a mountain landscape", opts)

# image1 and image2 will be identical
```

### High-Quality Generation

```elixir
# Use FLUX Dev for artistic quality
{:ok, image} = Margarine.generate("a cyberpunk cityscape",
  model: :flux_dev,
  steps: 28,
  guidance_scale: 3.5,
  size: {1024, 1024}
)

# Use SDXL Base for photorealistic quality
{:ok, image} = Margarine.generate("a photorealistic portrait",
  model: :sdxl_base,
  steps: 20,
  guidance_scale: 7.5,
  size: {1024, 1024}
)
```

### Image-to-Image Transformation

```elixir
# Subtle style change (keep most of original)
{:ok, image} = Margarine.img2img(
  "watercolor painting style",
  "photo.png",
  denoising_strength: 0.3,
  model: :sdxl_base
)

# Moderate transformation
{:ok, image} = Margarine.img2img(
  "oil painting, impressionist style",
  "photo.png",
  denoising_strength: 0.6,
  model: :flux_schnell
)

# Heavy modification (only rough composition remains)
{:ok, image} = Margarine.img2img(
  "cyberpunk city at night",
  "photo.png",
  denoising_strength: 0.8,
  model: :sdxl_base
)
```

### Error Handling

```elixir
case Margarine.generate(prompt, opts) do
  {:ok, image} ->
    Margarine.Image.save(image, "output.png")
    IO.puts("Image generated successfully!")

  {:error, reason} ->
    IO.puts("Generation failed: #{reason}")
end
```

### Health Checks

```elixir
# Check if Python environment is ready
env = Margarine.check_environment()

case env do
  %{pythonx_initialized: true, python_version: version} ->
    IO.puts("Ready! Python #{version}")

  %{pythonx_initialized: false} ->
    IO.puts("Python not initialized yet")
end
```

## Configuration Options

### Text-to-Image Options (`Margarine.generate/2`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:model` | `:flux_schnell` \| `:flux_dev` \| `:sdxl_turbo` \| `:sdxl_base` | `:flux_schnell` | Model to use |
| `:steps` | `pos_integer()` | Model-specific | Number of denoising steps (4 for schnell, 28 for dev, 1 for turbo, 20 for base) |
| `:guidance_scale` | `float()` | Model-specific | Guidance strength (0.0 for schnell/turbo, 3.5 for dev, 7.5 for base) |
| `:seed` | `integer()` \| `nil` | `nil` (random) | Random seed for reproducibility |
| `:size` | `{height, width}` | `{1024, 1024}` | Image dimensions (must be divisible by 8) |

### Image-to-Image Options (`Margarine.img2img/3`)

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:model` | `:flux_schnell` \| `:flux_dev` \| `:sdxl_turbo` \| `:sdxl_base` | `:flux_schnell` | Model to use |
| `:denoising_strength` | `float()` (0.0-1.0) | `0.75` | How much to modify (0.0 = no change, 1.0 = complete regeneration) |
| `:steps` | `pos_integer()` | Model-specific | Number of denoising steps |
| `:guidance_scale` | `float()` | Model-specific | Guidance strength |
| `:seed` | `integer()` \| `nil` | `nil` (random) | Random seed for reproducibility |
| `:size` | `{height, width}` \| `nil` | Auto-detect from image | Target dimensions (defaults to original image size, rounded to multiple of 8) |

## Architecture

Margarine uses a modular architecture:

- **Public API** (`Margarine`) - Clean Elixir interface
- **Pipeline** - Orchestrates generation workflow
- **Schedulers** - Pure Nx implementations (Euler, DDIM, etc.)
- **Python Server** - FLUX model inference via Pythonx
- **Image Module** - Nx tensor ↔ PNG/JPEG conversion
- **Memory Module** - Safety checks to prevent OOM

All Python dependencies are managed automatically via UV. No manual `pip install` or virtualenv management required!

## Development

This project follows strict **Test-Driven Development (TDD)** practices.

### Running Tests

```bash
# Fast unit tests (default)
mix test

# With coverage report
mix test --cover

# Integration tests (requires model download, ~12GB)
mix test --only integration

# All tests including integration
mix test --include integration

# Linting
mix credo --strict
```

### Test Coverage

We maintain **80%+ code coverage**. Current coverage: **75.3%**

### Contributing

We welcome contributions! Please:
1. Write tests first (TDD)
2. Ensure all tests pass (`mix test`)
3. Run Credo (`mix credo --strict`)
4. Update documentation as needed

## Roadmap

### Phase 1: FLUX MVP ✅ (v0.1.0)
- [x] FLUX Schnell and Dev support
- [x] Zero-copy Pythonx integration
- [x] Automatic Python/dependency installation
- [x] Memory safety checks
- [x] Comprehensive test suite

### Phase 2: SDXL + Image-to-Image ✅ (v0.2.0)
- [x] SDXL Base and Turbo support
- [x] Dual CLIP text encoders
- [x] DDIM scheduler implementation
- [x] Image-to-image for both FLUX and SDXL
- [x] Automatic dimension handling (RGBA, non-square, auto-rounding)
- [x] Comprehensive img2img tests

### Phase 3: Inpainting (Planned)
- [ ] SDXL Inpaint model support
- [ ] Mask preprocessing utilities
- [ ] Inpainting pipeline
- [ ] Interactive mask editing

### Phase 4: Advanced Features (Future)
- [ ] Streaming intermediate results
- [ ] Batch generation
- [ ] LoRA support
- [ ] ControlNet integration
- [ ] Custom schedulers (DPM++, etc.)
- [ ] Performance optimizations

## Troubleshooting

### "Python environment not initialized"

The first run takes 2-5 minutes to download Python and dependencies. Subsequent runs are instant.

### Out of Memory Errors

- Reduce image size: `size: {512, 512}`
- Use FLUX Schnell instead of Dev
- Close other memory-intensive applications
- Check available memory: `Margarine.Memory.check_available()`

### Slow Generation

- Ensure you're using GPU acceleration (EMLX for Apple Silicon, EXLA for NVIDIA)
- Use FLUX Schnell (4 steps) instead of Dev (28 steps)
- First run downloads models (~12GB) which takes time

## Documentation

- **Interactive Tutorials**: See the [Interactive Tutorials](#-interactive-tutorials-recommended) section above
  - [`notebooks/flux_getting_started.livemd`](notebooks/flux_getting_started.livemd) - FLUX tutorial
  - [`notebooks/sdxl_getting_started.livemd`](notebooks/sdxl_getting_started.livemd) - SDXL tutorial
- **API Docs**: [hexdocs.pm/margarine](https://hexdocs.pm/margarine)
- **Examples**: See `examples/` directory for runnable scripts
- **Changelog**: See `CHANGELOG.md` for version history

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- **FLUX** - Black Forest Labs for the amazing FLUX models
- **Pythonx** - For enabling seamless Python integration
- **Nx** - The Elixir numerical computing foundation
- **Bumblebee** - Inspiration for ML in Elixir

---

Built with ❤️ by the Elixir community. Logo generated with FLUX - a perfect example of what this library can do!
