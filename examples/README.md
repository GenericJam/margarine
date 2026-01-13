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

### `basic.exs` - Simple Image Generation

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

### `advanced.exs` - Advanced Features

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

## Tips

### Faster Generation

- Use FLUX Schnell instead of Dev (4 steps vs 28)
- Reduce image size: `size: {512, 512}`
- Use GPU acceleration (EMLX or EXLA)

### Higher Quality

- Use FLUX Dev: `model: :flux_dev`
- Increase steps (but slower): `steps: 28`
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
