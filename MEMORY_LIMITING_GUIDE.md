# Memory Limiting Guide for Margarine

## The Problem: OOM Crashes During Tests

When running integration tests with large ML models (FLUX, SDXL), Python processes can consume all available system memory, causing:
- System-wide freezing
- OOM (Out of Memory) kills
- Kernel panics on macOS
- Test failures and crashes

## Our Multi-Layered Defense Strategy

We use **three layers** of memory protection:

### 1. Elixir-Side Memory Checks (Primary Defense)
**File:** `lib/margarine/memory.ex`

**What it does:**
- Checks available system memory before loading models
- Compares against model memory estimates (FLUX: 14-26GB)
- Fails gracefully with clear error messages
- Works on all platforms (macOS, Linux, Windows)

**Example:**
```elixir
# In PythonxServer.handle_continue(:load_model)
required_mb = Margarine.Memory.estimate_flux_memory(:flux_schnell)  # 14GB
{:ok, info} = Margarine.Memory.available_memory()
available_mb = Margarine.Memory.bytes_to_mb(info.available)

if available_mb < required_mb do
  {:stop, {:insufficient_memory, "Need #{required_mb}MB, have #{available_mb}MB"}, state}
end
```

### 2. Sequential Test Execution (Secondary Defense)
**File:** `test/integration/flux_generation_test.exs`

**What it does:**
- Tests run sequentially with `async: false`
- Prevents multiple model instances loading simultaneously
- Shares single PythonxServer instance via `setup_all`
- Critical for machines with <32GB RAM

**Example:**
```elixir
# Integration tests MUST NOT run in parallel
use ExUnit.Case, async: false  # ← CRITICAL!

@describetag :integration
@describetag timeout: 300_000  # 5 minutes per test
```

### 3. Python-Side Memory Limits (Tertiary Defense - Linux Only)
**File:** `priv/python/memory_limit.py`

**What it does:**
- **On macOS:** DISABLED (see below)
- **On Linux:** Sets `RLIMIT_DATA` to cap heap allocations
- Auto-applied when module is imported
- Configurable via `set_memory_limit_gb(N)`

**Why not macOS?**
See next section.

## Why RLIMIT_AS Doesn't Work with Pythonx on macOS

### The Technical Issue

**Pythonx runs Python *in-process* via NIFs (Native Implemented Functions):**
1. Python is embedded in the same process as the BEAM
2. They share the same virtual address space
3. BEAM's address space is already >16GB (includes all libraries, mmap'd files, etc.)
4. Setting `RLIMIT_AS` to 16GB fails: `"current limit exceeds maximum"`

### The Failure Mode

```python
# This fails on macOS when run via Pythonx:
import resource
limit_bytes = 16 * 1024**3  # 16GB
resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
# OSError: current limit exceeds maximum limit
```

**Result:** Process crashes or becomes unstable, causing:
- stderr write failures (`"device does not exist"`)
- BEAM crashes
- Erl crash dumps

### The Solution

**Don't use `RLIMIT_AS` on macOS. Use `RLIMIT_DATA` on Linux instead.**

```python
import platform

if platform.system() == "Darwin":  # macOS
    print("Skipping memory limiting (embedded Python)")
    return  # Rely on Elixir-side checks

# Linux only
if hasattr(resource, 'RLIMIT_DATA'):
    # RLIMIT_DATA limits heap allocations (good)
    # RLIMIT_AS limits address space (bad for embedded Python)
    resource.setrlimit(resource.RLIMIT_DATA, (limit_bytes, limit_bytes))
```

## Why This Strategy Works

### Layer 1: Elixir Checks (Works Everywhere)
✅ Runs before Python starts
✅ Platform-agnostic (macOS, Linux, Windows)
✅ Uses system APIs (vm_stat, /proc/meminfo, wmic)
✅ Graceful failures with helpful messages

### Layer 2: Sequential Tests (Prevents Parallel OOM)
✅ Single model instance at a time
✅ Prevents combinatorial memory explosion
✅ Cleanup between tests (`on_exit` callback)
✅ Shared server instance (`setup_all`)

### Layer 3: Python Limits (Linux Defense-in-Depth)
✅ Additional safety net on Linux
✅ Limits heap allocations (not address space)
✅ Doesn't interfere with mmap'd model weights
✅ Skipped on macOS to avoid crashes

## Test Running Best Practices

### Run Non-Integration Tests (Fast, Safe)
```bash
# Default - excludes integration tests
mix test

# With coverage
mix test --cover
```

### Run Integration Tests (Sequential, 16GB+ RAM)
```bash
# Only integration tests
mix test --only integration

# All tests including integration
mix test --include integration
```

### Monitor Memory While Tests Run
```bash
# Terminal 1: Run tests
mix test --only integration

# Terminal 2: Watch memory usage
watch -n 1 'ps aux | grep python | head -5'
```

### Clean Up Crashed Processes
```bash
# Kill any stuck Python processes
pkill -f python

# Clean up build artifacts
mix clean

# Remove crash dump
rm erl_crash.dump
```

## Production Deployment Considerations

### macOS (Development)
- ✅ Elixir memory checks are sufficient
- ✅ No Python-side limiting needed
- ⚠️ Watch Activity Monitor during large jobs
- ⚠️ Don't run multiple model instances simultaneously

### Linux (Production)
- ✅ All three layers active
- ✅ `RLIMIT_DATA` provides additional safety
- ✅ Consider Docker memory limits for extra protection
- ✅ Use systemd resource limits in production

```ini
# /etc/systemd/system/margarine.service
[Service]
MemoryMax=32G
MemoryHigh=28G
```

### Docker
```dockerfile
# Dockerfile
FROM elixir:1.19

# Set memory limits at container level
# docker run --memory=32g --memory-swap=32g margarine
```

## Configuring Memory Limits

### Increase Limit for Multiple Models
```python
# priv/python/flux_pythonx.py or sdxl_pythonx.py
from memory_limit import set_memory_limit_gb

# For 2x FLUX instances (32GB heap on Linux)
set_memory_limit_gb(32)
```

### Decrease Limit for Smaller Machines
```python
# For 8GB heap limit (testing, small models)
set_memory_limit_gb(8)
```

### Disable Entirely
```python
# priv/python/memory_limit.py
if __name__ != "__main__":
    # Comment out to disable:
    # set_memory_limit_gb(16)
    pass
```

## Troubleshooting

### Symptom: "device does not exist" crash
**Cause:** RLIMIT_AS failing on macOS
**Fix:** Update to latest `memory_limit.py` (should skip macOS)

### Symptom: OOM kills during tests
**Cause:** Tests running in parallel
**Fix:** Ensure `async: false` in integration tests

### Symptom: "Insufficient memory" before model load
**Cause:** Elixir memory check detecting low memory
**Fix:** Close other apps or use smaller model (`:flux_schnell` vs `:flux_dev`)

### Symptom: Slow test performance
**Cause:** Normal - integration tests load 12-26GB models
**Fix:** Use fast tests (`mix test`) for development, integration for CI

## References

- **Python resource module:** https://docs.python.org/3/library/resource.html
- **Pythonx architecture:** https://github.com/cocoa-xu/pythonx
- **FLUX models:** https://huggingface.co/black-forest-labs
- **Similar issue (ml-stable-diffusion):** https://github.com/apple/ml-stable-diffusion/issues/8

## Summary

**✅ Memory protection strategy is working correctly:**
1. Elixir checks prevent loading on low-memory systems
2. Sequential tests prevent parallel OOM
3. Python limits (Linux only) provide defense-in-depth
4. macOS skips Python limits (would crash with Pythonx)

**⚠️ If you see crashes:**
- Check `async: false` in integration tests
- Ensure latest `memory_limit.py` (skips macOS)
- Monitor memory during test runs
- Use `mix test` (not `mix test --include integration`) for development
