# EXLA Setup Guide for Linux (NVIDIA GPU)

This guide walks you through setting up EXLA with CUDA support on Linux for GPU-accelerated image generation with Margarine.

## Prerequisites

- NVIDIA GPU with compute capability 5.0 or higher
- Ubuntu/Debian-based Linux distribution (Pop!_OS, Ubuntu 22.04+, etc.)
- At least 16GB RAM for SDXL models

## Step 1: Install System Dependencies

Install the required build tools and development packages:

```bash
sudo apt-get update
sudo apt-get install -y build-essential erlang-dev
```

## Step 2: Install CUDA Toolkit

EXLA requires CUDA 12.x. We recommend CUDA 12.8 for best compatibility.

### 2.1 Add NVIDIA CUDA Repository

```bash
# Download and install the CUDA keyring
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb

# Update package lists
sudo apt-get update
```

### 2.2 Install CUDA Toolkit 12.8

```bash
sudo apt-get -y install cuda-toolkit-12-8
```

**Note:** This downloads ~3GB and may take several minutes.

### 2.3 Verify CUDA Installation

```bash
ls -la /usr/local/cuda-12.8/bin/nvcc
```

You should see the nvcc compiler binary.

## Step 3: Install NCCL

NCCL (NVIDIA Collective Communications Library) is required by EXLA for GPU operations:

```bash
sudo apt-get install -y libnccl2 libnccl-dev
```

Verify installation:

```bash
ldconfig -p | grep nccl
```

You should see `libnccl.so.2` in the output.

## Step 4: Install cuDNN

cuDNN is required for deep learning operations. Install version 9.x for CUDA 12:

```bash
sudo apt-get install -y libcudnn9-cuda-12 libcudnn9-dev-cuda-12
```

## Step 5: Configure Environment Variables

Add CUDA paths and XLA configuration to your shell configuration file.

### For Zsh (default on Pop!_OS and modern Ubuntu)

Edit `~/.zshrc`:

```bash
# CUDA environment variables
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
export XLA_FLAGS=--xla_gpu_cuda_data_dir=/usr/local/cuda-12.8
export XLA_TARGET=cuda12
export CUDA_VISIBLE_DEVICES=0  # Tells PyTorch/Margarine to use GPU 0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # Reduce memory fragmentation
```

### For Bash

Edit `~/.bashrc` with the same variables above.

### Apply Changes

```bash
# For Zsh
source ~/.zshrc

# For Bash
source ~/.bashrc
```

### Verify Environment

```bash
echo "XLA_TARGET: $XLA_TARGET"
echo "CUDA in PATH:"
which nvcc
nvcc --version
```

You should see:
- `XLA_TARGET: cuda12`
- nvcc path: `/usr/local/cuda-12.8/bin/nvcc`
- CUDA version: `12.8`

**Important:** Make sure `XLA_BUILD` is NOT set:

```bash
echo "XLA_BUILD: $XLA_BUILD"
```

This should be empty. If it shows "true", unset it:

```bash
unset XLA_BUILD
```

## Step 6: Add EXLA to Your Mix Project

Edit `mix.exs` and add EXLA to your dependencies:

```elixir
def deps do
  [
    {:margarine, "~> 0.2.0"},
    {:exla, "~> 0.10.0"}
  ]
end
```

## Step 7: Install Dependencies

**Important:** Open a fresh terminal (or source your shell config) before running these commands to ensure environment variables are set.

```bash
cd your_project
mix deps.get
mix deps.compile
```

The compilation will:
1. Download precompiled XLA binaries for CUDA 12 (~200MB)
2. Compile EXLA with CUDA support
3. You should see: "CUDA is available." during compilation

If you see "CUDA is not available", check:
- Environment variables are set correctly (`echo $XLA_TARGET`)
- nvcc is accessible (`which nvcc`)
- You're in a fresh terminal session with sourced shell config

## Step 8: Configure EXLA Backend in Your Application

### For Scripts (using Mix.install)

If using `Mix.install` in scripts, you cannot run them with `elixir` directly. Use `mix run` instead:

```elixir
# In your_script.exs
Application.ensure_all_started(:margarine)

# Configure EXLA backend for NVIDIA GPU
Application.put_env(:nx, :default_backend, EXLA.Backend)

# Configure EXLA clients
Application.put_env(:exla, :clients,
  cuda: [platform: :cuda],
  host: [platform: :host]
)

# Set preferred clients to only use cuda
Application.put_env(:exla, :preferred_clients, [:cuda, :host])

# Set cuda as the default client
Application.put_env(:exla, :default_client, :cuda)
```

Run with:

```bash
mix run your_script.exs
```

### For Mix Projects

Create or edit `config/config.exs`:

```elixir
import Config

# Configure Nx to use EXLA backend
config :nx, default_backend: EXLA.Backend

# Configure EXLA clients
config :exla,
  clients: [
    cuda: [platform: :cuda],
    host: [platform: :host]
  ],
  preferred_clients: [:cuda, :host],
  default_client: :cuda
```

### For Livebook

Add a setup cell at the top of your notebook:

```elixir
# Configure EXLA backend for NVIDIA GPU
Nx.global_default_backend(EXLA.Backend)

Application.put_env(:exla, :clients,
  cuda: [platform: :cuda],
  host: [platform: :host]
)

Application.put_env(:exla, :preferred_clients, [:cuda, :host])
Application.put_env(:exla, :default_client, :cuda)
```

**Important:** Restart Livebook after setting up environment variables in your shell config.

## Step 9: Verify GPU Acceleration

Run a simple test to verify EXLA is using your GPU:

```elixir
# In iex -S mix or a Livebook cell
tensor = Nx.tensor([[1, 2], [3, 4]], backend: EXLA.Backend)
IO.inspect(Nx.backend_transfer(tensor))
```

Or check available clients:

```elixir
EXLA.Client.get_supported_platforms()
# Should include :cuda
```

## Troubleshooting

### "CUDA is not available" during compilation

**Cause:** nvcc not found in PATH or XLA_TARGET not set.

**Solution:**
1. Verify environment variables: `echo $XLA_TARGET` (should be `cuda12`)
2. Verify nvcc is accessible: `which nvcc`
3. Open a fresh terminal and source your shell config
4. Clean and recompile:
   ```bash
   mix deps.clean exla xla --build
   rm -rf ~/.cache/xla
   mix deps.compile
   ```

### "unknown client :rocm" or ":gpu" errors

**Cause:** Incorrect client configuration.

**Solution:** Ensure you set `preferred_clients` to only include `:cuda` and `:host`:

```elixir
Application.put_env(:exla, :preferred_clients, [:cuda, :host])
```

### "failed to extract xla archive" with "build" directory

**Cause:** `XLA_BUILD=true` is set, forcing local compilation instead of using precompiled binaries.

**Solution:**

```bash
unset XLA_BUILD
mix deps.clean exla xla --build
rm -rf ~/.cache/xla
mix deps.compile
```

### XLA downloads CPU version instead of CUDA version

**Cause:** Environment variables not set when Mix runs.

**Solution:**
1. Verify in the same terminal: `echo $XLA_TARGET` (must show `cuda12`)
2. Make sure you sourced your shell config in the current session
3. Don't run commands through other tools that might not preserve environment

### CUDA Out of Memory errors

**Cause:** GPU doesn't have enough VRAM for the model at the requested resolution.

**Solutions for 4GB GPUs (like RTX 3050 Ti):**

1. **Use smaller image sizes:**
   ```elixir
   # Instead of 1024x1024 (default), use 512x512 or 768x768
   {:ok, image} = Margarine.generate("prompt",
     model: :sdxl_turbo,
     width: 512,
     height: 512
   )
   ```

2. **Verify memory optimization is enabled:**
   ```bash
   echo $PYTORCH_CUDA_ALLOC_CONF  # Should show: expandable_segments:True
   ```

3. **Monitor GPU memory usage:**
   ```bash
   nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv -l 1
   ```

4. **For persistent issues:**
   - Use 512x512 resolution (uses 1/4 the VRAM of 1024x1024)
   - Stick with SDXL Turbo (uses less VRAM than SDXL Base)
   - Consider CPU inference for larger sizes (slower but no VRAM limit)

See [MEMORY_LIMITING_GUIDE.md](MEMORY_LIMITING_GUIDE.md) for more memory management strategies.

## Verifying Successful Setup

When everything is configured correctly:

1. `mix deps.compile exla` should show:
   - "Downloading a precompiled XLA archive for target x86_64-linux-gnu-**cuda12**"
   - "CUDA is available."

2. Running Margarine should use GPU:
   ```elixir
   {:ok, image} = Margarine.generate("test image", model: :sdxl_turbo, steps: 1)
   ```

3. Check GPU usage during generation:
   ```bash
   # In another terminal
   watch -n 1 nvidia-smi
   ```
   You should see GPU memory usage increase and GPU utilization during image generation.

## Performance Notes

- First run downloads models (~7GB for SDXL), which takes 2-5 minutes
- Subsequent runs are much faster
- GPU memory usage (at 1024x1024 resolution):
  - SDXL Turbo: ~4-5GB VRAM (use 512x512 for 4GB GPUs)
  - SDXL Base: ~6GB VRAM (requires 6GB+ GPU)
  - FLUX models: ~8GB+ VRAM (requires 8GB+ GPU)
- Image resolution affects VRAM usage:
  - 512x512: ~1-2GB VRAM (works on 4GB GPUs)
  - 768x768: ~2-3GB VRAM (works on 4GB GPUs)
  - 1024x1024: ~4-5GB VRAM (needs 6GB+ GPU for stability)

## System Requirements Summary

- **Minimum VRAM:** 4GB (for SDXL Turbo at 512x512 resolution)
- **Recommended VRAM:** 6GB+ (for SDXL at 1024x1024), 8GB+ (for FLUX models)
- **RAM:** 16GB minimum
- **CUDA:** 12.1 or higher (12.8 recommended)
- **cuDNN:** 9.1 or higher
- **NCCL:** 2.x
- **NVIDIA Driver:** 580.x or higher (check with `nvidia-smi`)

**For 4GB GPUs:** Use smaller resolutions (512x512 or 768x768) to avoid out-of-memory errors.

## Additional Resources

- [EXLA Documentation](https://hexdocs.pm/exla/)
- [XLA GitHub Repository](https://github.com/elixir-nx/xla)
- [NVIDIA CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/)
- [Margarine README](README.md)
