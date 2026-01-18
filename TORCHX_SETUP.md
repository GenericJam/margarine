# Torchx Setup Guide for Margarine

**🧪 EXPERIMENTAL - Promising but not extensively tested**

This guide explains how to use Torchx as an Nx backend with Margarine. Torchx is an experimental alternative to EMLX that offers cross-platform compatibility and comparable performance on Apple Silicon.

## When to Consider Torchx

Torchx may be appropriate if you:
- Need **cross-platform development**: Same codebase on Mac, Linux, and Windows
- Want **PyTorch ecosystem compatibility**: Already using PyTorch for other projects
- Are experimenting with different backends
- Need flexibility to switch between CPU and GPU easily

## When to Use Other Backends

**Consider EMLX if:**
- You're on **Apple Silicon** and want the most tested/stable option
- You prefer the officially recommended backend for macOS

**Consider EXLA if:**
- You're on **Linux with NVIDIA GPU** (best performance on CUDA)
- You need XLA-specific optimizations

## Performance Comparison

**Benchmark on Apple M4 Max (64GB):**
- Test: FLUX Schnell, 1024x1024, 4 steps, seed 42
- EMLX (MPS): 1m 34.6s
- Torchx (MPS): 1m 27.9s (**7% faster**)

**Key findings:**
- ✅ Torchx with MPS performs comparably to EMLX (~7% difference)
- ✅ Both backends suitable for production on this hardware
- 🧪 This is a recent discovery - Torchx may have edge cases not yet tested

| Backend | Device | Speed (1024x1024) | Maturity |
|---------|--------|-------------------|----------|
| **EMLX** | MPS (GPU) | ~1m 35s | ✅ Stable |
| **Torchx** | MPS (GPU) | ~1m 28s | 🧪 Experimental |
| **Torchx** | CPU | Not tested | ⚠️ Likely slow |
| **EXLA** | CUDA | ~2 min | ✅ Stable (Linux) |

## Approaches

### Recommended: MPS (Apple Silicon GPU)

Torchx can use MPS on Apple Silicon with comparable performance to EMLX.

---

## Setup Instructions

### Prerequisites

- Elixir 1.14+
- At least 16GB RAM
- Patience (CPU inference is very slow)

### Step 1: Add Torchx to Your Project

**For Mix projects**, edit `mix.exs`:

```elixir
def deps do
  [
    {:margarine, "~> 0.2.0"},
    {:torchx, "~> 0.10"}
  ]
end
```

**For Mix.install scripts**, see Step 3 below.

### Step 2: Install Dependencies

```bash
cd your_project
mix deps.get
mix deps.compile
```

**What happens:**
- Torchx downloads `libtorch-2.8.0-cpu` (~200MB)
- Compiles native extensions using CMake
- May take 2-5 minutes on first install

**If compilation fails with "cmake not found":**
- macOS: `brew install cmake`
- Linux: `sudo apt-get install cmake`

### Step 3: Configure Backends

Configure Torchx to use MPS (Apple Silicon GPU) or CPU.

#### For Mix Projects

Create or edit `config/config.exs`:

```elixir
import Config

# Configure Nx to use Torchx backend
# Use :mps for Apple Silicon GPU, :cpu for CPU-only
config :nx, default_backend: {Torchx.Backend, device: :mps}
```

#### For Mix.install Scripts

**For Apple Silicon (MPS):**
```elixir
Mix.install(
  [
    {:margarine, "~> 0.2.0"},
    {:torchx, "~> 0.10"}
  ],
  config: [
    nx: [default_backend: {Torchx.Backend, device: :mps}]
  ]
)

# Match Python device to Nx backend
System.put_env("MARGARINE_DEVICE", "mps")
```

**For CPU-only:**
```elixir
Mix.install(
  [
    {:margarine, "~> 0.2.0"},
    {:torchx, "~> 0.10"}
  ],
  config: [
    nx: [default_backend: {Torchx.Backend, device: :cpu}]
  ]
)

System.put_env("MARGARINE_DEVICE", "cpu")
```

#### For Livebook

Add a setup cell:

```elixir
# Setup cell - Use MPS for Apple Silicon
System.put_env("MARGARINE_DEVICE", "mps")

Mix.install(
  [
    {:margarine, "~> 0.2.0"},
    {:torchx, "~> 0.10"},
    {:kino, "~> 0.14"}
  ],
  config: [
    nx: [default_backend: {Torchx.Backend, device: :mps}]
  ]
)
```

### Step 4: Verify Setup

Test that Torchx is working:

```elixir
# In iex -S mix or a Livebook cell
tensor = Nx.tensor([[1, 2], [3, 4]], backend: {Torchx.Backend, device: :cpu})
IO.inspect(tensor)
# Should show: Torchx.Backend(cpu)
```

### Step 5: Generate an Image

```elixir
{:ok, image} = Margarine.generate(
  "a serene mountain landscape at sunset",
  model: :flux_schnell,
  steps: 4,
  size: {1024, 1024},
  seed: 42
)

Margarine.Image.save(image, "output_torchx.png")
```

**Expected time (MPS on M4 Max):**
- 1024x1024, FLUX Schnell, 4 steps: ~1m 30s
- Similar to EMLX performance (~7% difference in benchmarks)

---

## Troubleshooting

### "cmake not found in the path"

**On macOS:**
```bash
brew install cmake
```

**On Linux:**
```bash
sudo apt-get install cmake
```

**In Livebook:** Livebook doesn't inherit your PATH. Add to setup cell:
```elixir
System.put_env("PATH", "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin")
```

### "PyTorch is not linked with support for mps/cuda devices"

**Cause:** Device mismatch between Python device and Torchx backend device.

**Solution:** Ensure both sides use the same device:

```elixir
# Both must match
System.put_env("MARGARINE_DEVICE", "mps")  # Python side
# AND
config: [nx: [default_backend: {Torchx.Backend, device: :mps}]]  # Elixir side
```

### "Could not compile dependency :telemetry" (Livebook)

**Cause:** Version conflicts between Livebook's bundled dependencies.

**Solutions:**

1. **Use Livebook Server** (recommended):
   ```bash
   mix escript.install hex livebook
   livebook server
   ```

2. **Force clean install**:
   ```elixir
   Mix.install([...], force: true)
   ```

3. **Pin telemetry version**:
   ```elixir
   Mix.install([
     {:margarine, "~> 0.2.0"},
     {:torchx, "~> 0.10"},
     {:telemetry, "~> 1.2.1", override: true}
   ])
   ```

### Slow Generation

**If using CPU device:** This is expected - CPU inference is much slower than GPU.

**Solution:** Use MPS device on Apple Silicon:
```elixir
System.put_env("MARGARINE_DEVICE", "mps")
config: [nx: [default_backend: {Torchx.Backend, device: :mps}]]
```

### Out of Memory

**On CPU, RAM usage is higher** than GPU inference.

**Solutions:**
- Close other applications
- Use smaller image sizes (512x512 instead of 1024x1024)
- Reduce steps
- Check available RAM: `Margarine.Memory.available_memory()`

---

## Environment Variables

### MARGARINE_DEVICE

**Purpose:** Override automatic device detection

**Values:**
- `"cpu"` - Force CPU (required for Torchx on macOS)
- `"mps"` - Use Apple Silicon GPU (not compatible with CPU-only Torchx)
- `"cuda"` - Use NVIDIA GPU (requires CUDA Torchx)

**Example:**
```bash
export MARGARINE_DEVICE=cpu
elixir examples/basic_torchx.exs
```

### LIBTORCH_TARGET

**Purpose:** Choose which libtorch distribution Torchx downloads

**Values:**
- `cpu` (default) - CPU-only version
- `cu118` - CUDA 11.8
- `cu126` - CUDA 12.6
- `cu128` - CUDA 12.8

**Note:** No MPS target available. MPS requires manual compilation.

**Example:**
```bash
# For NVIDIA GPU with CUDA 12.8
export LIBTORCH_TARGET=cu128
mix deps.compile torchx --force
```

### LIBTORCH_DIR

**Purpose:** Use a custom-compiled libtorch (for MPS support)

**Example:**
```bash
export LIBTORCH_DIR=/path/to/custom/libtorch
mix deps.compile torchx --force
```

**Not recommended:** This requires advanced knowledge of PyTorch compilation.

---

## Example Script

See [`examples/basic_torchx.exs`](examples/basic_torchx.exs) for a complete working example.

To run:
```bash
cd margarine
elixir examples/basic_torchx.exs
```

---

## FAQ

### Q: Can Torchx use Apple Silicon GPU (MPS)?

**A:** Yes! Set `device: :mps` in the Torchx backend config and `MARGARINE_DEVICE=mps`. Performance is comparable to EMLX (~7% difference in benchmarks).

### Q: Should I use Torchx or EMLX on Apple Silicon?

**A:** Both perform similarly. Choose based on needs:
- **EMLX**: More tested/stable, Apple-specific
- **Torchx**: Experimental but promising, cross-platform compatible

### Q: Is Torchx production-ready?

**A:** It's experimental. Performance is good, but it hasn't been extensively tested with Margarine. Use at your own risk in production. EMLX is the safer choice for now.

### Q: When should I actually use Torchx?

**A:** Use Torchx if you:
- Develop on Windows/Linux/Mac and want the same backend everywhere
- Already have PyTorch CUDA set up for other projects
- Need CPU-only fallback for testing
- Are on Linux with NVIDIA GPU and prefer PyTorch over XLA

### Q: What about CUDA support on Linux?

**A:** Torchx supports CUDA! Set `LIBTORCH_TARGET=cu128` before compiling. But for NVIDIA GPUs, we still recommend EXLA (see [EXLA Setup Guide](EXLA_SETUP_LINUX_NVIDIA.md)).

---

## Backend Recommendations by Platform

| Platform | Recommended | Experimental | Notes |
|----------|-------------|--------------|-------|
| **macOS (Apple Silicon)** | EMLX | Torchx (MPS) | Both ~similar performance |
| **Linux (NVIDIA GPU)** | EXLA | Torchx (CUDA) | EXLA more mature |
| **Linux (CPU only)** | EXLA | Torchx | - |
| **Windows** | Torchx | - | Only option currently |

---

## Additional Resources

- [Torchx GitHub Repository](https://github.com/elixir-nx/nx/tree/main/torchx)
- [Torchx Hex Documentation](https://hexdocs.pm/torchx/)
- [PyTorch MPS Documentation](https://pytorch.org/docs/stable/notes/mps.html)
- [Nx Backend Documentation](https://hexdocs.pm/nx/Nx.html#module-backends)
- [EMLX Setup (Recommended for macOS)](README.md#installation)
- [EXLA Setup (Recommended for Linux/NVIDIA)](EXLA_SETUP_LINUX_NVIDIA.md)

---

## Summary

**Torchx with Margarine (Experimental):**
- 🧪 Comparable performance to EMLX on Apple Silicon (~7% difference)
- 🧪 Works with MPS (Apple Silicon GPU) out of the box
- 🧪 Not extensively tested - may have edge cases
- ✅ Cross-platform compatible (Mac, Linux, Windows)
- ✅ PyTorch ecosystem integration
- ⚠️ Less mature than EMLX for Apple Silicon

**Recommendation:**
- **Experimenting/Testing:** Torchx is worth trying
- **Production (Apple Silicon):** EMLX is safer (more tested)
- **Production (Linux/NVIDIA):** EXLA is recommended
- **Cross-platform development:** Torchx is a good choice
