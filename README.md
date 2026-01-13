# Margarine

<p align="center">
  <strong>"I Can't Believe It's Not Butter... I Mean Python!"</strong>
</p>

---

AI-powered image generation for Elixir using FLUX and Stable Diffusion.

Margarine brings state-of-the-art text-to-image generation to the Elixir ecosystem with a clean, native API. Generate beautiful images from text prompts with just a few lines of code.

## Features

- 🎨 **FLUX Integration** - Fast, high-quality image generation with FLUX Schnell and FLUX Dev
- ⚡ **Zero-Copy Performance** - Pythonx integration for efficient tensor transfer between Elixir and Python
- 🚀 **Zero Configuration** - Python and dependencies installed automatically on first run via UV
- 🍎 **Apple Silicon Support** - Optimized for M-series Macs with EMLX/Metal backend
- 🔥 **CUDA Support** - NVIDIA GPU acceleration via EXLA/XLA
- 💾 **Memory Safety** - Automatic checks prevent OOM crashes
- 🧪 **Production Ready** - Comprehensive tests (119+), telemetry, and error handling
- 📊 **Type Safe** - Full Dialyzer type specs on all public functions

## Quick Start

```elixir
# Add to mix.exs
def deps do
  [
    {:margarine, "~> 0.1.0"},
    {:emlx, "~> 0.1"}  # For Apple Silicon
    # OR
    # {:exla, "~> 0.9"}  # For NVIDIA/AMD GPU or CPU
  ]
end
```

```elixir
# Simple text-to-image
{:ok, image} = Margarine.generate("a red panda eating bamboo")
Margarine.Image.save(image, "panda.png")

# Advanced options
{:ok, image} = Margarine.generate("a serene mountain landscape at sunset",
  model: :flux_schnell,  # or :flux_dev for higher quality
  steps: 4,              # 4 for schnell, 28 for dev
  seed: 42,              # for reproducibility
  size: {1024, 1024}     # width x height
)
```

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

On first run, Margarine will automatically:
1. Download and install Python 3.11+ via UV (~100MB)
2. Install PyTorch and dependencies (~500MB)
3. Cache everything for instant subsequent runs

**This takes 2-5 minutes on first run.** For production deployments, we recommend running a "warm-up" generation when your server starts:

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
# Use FLUX Dev for higher quality (slower)
{:ok, image} = Margarine.generate("a photorealistic portrait",
  model: :flux_dev,
  steps: 28,
  guidance_scale: 3.5,
  size: {1024, 1024}
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

All options are passed to `Margarine.generate/2`:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `:model` | `:flux_schnell` \| `:flux_dev` | `:flux_schnell` | Model to use |
| `:steps` | `pos_integer()` | Model-specific (4 or 28) | Number of denoising steps |
| `:guidance_scale` | `float()` | Model-specific (0.0 or 3.5) | Guidance strength |
| `:seed` | `integer()` \| `nil` | `nil` (random) | Random seed for reproducibility |
| `:size` | `{width, height}` | `{1024, 1024}` | Image dimensions (must be divisible by 8) |

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

### Phase 1: FLUX MVP ✅
- [x] FLUX Schnell and Dev support
- [x] Zero-copy Pythonx integration
- [x] Automatic Python/dependency installation
- [x] Memory safety checks
- [x] Comprehensive test suite

### Phase 2: Advanced FLUX (Upcoming)
- [ ] Streaming intermediate results
- [ ] Batch generation
- [ ] Custom schedulers (DDIM, DPM++)
- [ ] Performance optimizations

### Phase 3: Image-to-Image (Future)
- [ ] Image loading and preprocessing
- [ ] Strength parameter
- [ ] img2img pipeline

### Phase 4: Stable Diffusion (Future)
- [ ] SD 1.5, SD 2.1, SDXL support
- [ ] Unified API across models
- [ ] LoRA support
- [ ] ControlNet integration

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

- **API Docs**: [hexdocs.pm/margarine](https://hexdocs.pm/margarine) (coming soon)
- **Examples**: See `examples/` directory
- **Architecture**: See `CLAUDE.md` for development notes
- **Changelog**: See `CHANGELOG.md`

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- **FLUX** - Black Forest Labs for the amazing FLUX models
- **Pythonx** - For enabling seamless Python integration
- **Nx** - The Elixir numerical computing foundation
- **Bumblebee** - Inspiration for ML in Elixir

---

Built with ❤️ by the Elixir community. Logo generated with FLUX - a perfect example of what this library can do!
