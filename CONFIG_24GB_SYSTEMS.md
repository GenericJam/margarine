# Configuration for 24GB Systems

If you have a system with 24GB unified memory and are experiencing timeout errors, use this configuration.

## Recommended Configuration

Create or update your `config/config.exs`:

```elixir
import Config

# Increase timeout for slower generation on memory-constrained systems
config :margarine,
  timeout: 900_000  # 15 minutes (default is 5 minutes)

# Configure your Nx backend
# For Apple Silicon (M1/M2/M3)
config :nx,
  default_backend: EMLX.Backend,
  default_defn_options: [compiler: EMLX]

# For NVIDIA GPU
# config :nx,
#   default_backend: EXLA.Backend,
#   default_defn_options: [compiler: EXLA]
```

## Recommended Models and Settings

### Best Choice: SDXL (7GB models)

SDXL models are smaller and more likely to work on 24GB systems:

```elixir
# SDXL Turbo - Fast, photorealistic (1 step)
{:ok, image} = Margarine.generate(
  "a serene mountain landscape at sunset",
  model: :sdxl_turbo,
  steps: 1,
  size: {1024, 1024}
)

# SDXL Base - Higher quality (20 steps)
{:ok, image} = Margarine.generate(
  "a serene mountain landscape at sunset",
  model: :sdxl_base,
  steps: 20,
  size: {1024, 1024}
)
```

### FLUX Schnell with Reduced Size (14GB model)

If you want to use FLUX, reduce the image size:

```elixir
# Start small
{:ok, image} = Margarine.generate(
  "a red panda eating bamboo",
  model: :flux_schnell,
  steps: 4,
  size: {512, 512}  # Half resolution = much less memory
)

# If 512x512 works, try 768x768
{:ok, image} = Margarine.generate(
  "a red panda eating bamboo",
  model: :flux_schnell,
  steps: 4,
  size: {768, 768}
)
```

## Memory Monitoring

Check available memory before generation:

```elixir
{:ok, info} = Margarine.Memory.check_available()
available_gb = info.available_bytes / 1_073_741_824
IO.puts("Available memory: #{Float.round(available_gb, 1)}GB")

# Only proceed if you have enough
if available_gb >= 18 do
  {:ok, image} = Margarine.generate("your prompt here")
else
  IO.puts("Not enough memory available. Close other applications.")
end
```

## Expected Performance

On a 24GB system:

- **SDXL Turbo @ 1024×1024**: ✅ Should work (1-2 minutes)
- **SDXL Base @ 1024×1024**: ✅ Should work (5-8 minutes)
- **FLUX Schnell @ 512×512**: ✅ Should work (2-4 minutes)
- **FLUX Schnell @ 768×768**: ⚠️ May work (4-7 minutes)
- **FLUX Schnell @ 1024×1024**: ❌ Likely to fail or OOM
- **FLUX Dev**: ❌ Will not work (requires 26GB)

## Troubleshooting

If you still get timeout errors after increasing the timeout:

1. **Close all other applications** to free up memory
2. **Restart your machine** to clear memory caches
3. **Try SDXL instead of FLUX** (smaller models)
4. **Reduce image size** to 512×512 or smaller
5. **Check memory usage** with Activity Monitor (macOS) or htop (Linux)

## First-Time Setup

The first run will download models (this only happens once):

- **SDXL**: ~7GB download (10-20 minutes depending on internet)
- **FLUX Schnell**: ~12GB download (15-30 minutes)

Be patient on first run! Subsequent runs will be much faster.
