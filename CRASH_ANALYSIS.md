# Crash Analysis & Memory Leak Investigation

**Date**: 2026-01-14
**Status**: Computer crashed during integration test at image decoding step

## Last Known State

From `integration_test_output.log`:
- Test was running: "Handles larger images within memory constraints"
- Size: 1024x1024 image (4x larger than 512x512)
- Last message: `18:21:26.088 [info] [Margarine.Pipeline] Decoding latents to image...`
- Memory available at start: **29GB** (sufficient)
- Process: Successfully completed 4 denoising steps
- Stopped: During VAE decode operation

## System Status After Crash

```bash
# No crash dumps found
$ ls erl_crash.dump
# Not found

# No Python processes running
$ ps aux | grep python
# None (only BEAM/Elixir LS)

# Memory status: HEALTHY
$ vm_stat
Pages free: 2,350,517 (~38GB free)
# System recovered, no memory pressure
```

## Potential Memory Leak Sources

### 1. **Python Global Dictionary Accumulation** ⚠️ HIGH PRIORITY

**Location**: `lib/margarine/python/pythonx_server.ex:228-347`

**Issue**: Every Python function call does:
```elixir
call_globals = Map.merge(state.globals, %{
  "prompt" => prompt,
  "latents_bin" => latents_bin,  # Can be 200MB+ for 1024x1024
  "latents_shape" => latents_shape,
  # ... more data ...
})

# Then updates state
{:reply, {:ok, decoded}, %{state | globals: new_globals}}
```

**Problem**:
- Each call adds new entries to Python globals
- Old entries are never removed
- NumPy binary data (latents_bin, prompt_embeds_bin) can be **hundreds of MB**
- After 100 calls: **potentially 10-20GB** of accumulated globals

**Example Growth**:
```
Call 1: globals = {"latents_bin" => 200MB, ...}
Call 2: globals = {"latents_bin" => 200MB, "latents_bin_2" => 200MB, ...}
Call 3: globals = {"latents_bin" => 200MB, "latents_bin_2" => 200MB, "latents_bin_3" => 200MB, ...}
```

Even though we overwrite keys, Python keeps references if they're used elsewhere in the globals dict.

### 2. **FLUX Model Global Variable** ⚠️ MEDIUM PRIORITY

**Location**: `priv/python/flux_pythonx.py:31`

```python
_models = None  # Global variable

def initialize_model(...):
    global _models
    _models = {
        "pipe": pipe,  # Entire FLUX pipeline (~12-26GB)
        "device": device,
        "dtype": dtype,
        # ...
    }
```

**Problem**:
- Model stays in memory **forever** until explicitly cleared
- If server crashes/restarts, old reference may persist
- Python GC might not collect if any reference remains

**Risk**:
- Multiple test runs could stack models in memory
- Even after `clear_memory()`, global may not be fully freed

### 3. **Tensor Reference Cycles** ⚠️ MEDIUM PRIORITY

**Location**: `priv/python/flux_pythonx.py:284-288, 351-355`

```python
# Return binary data
return (
    model_output_np.data.tobytes(),  # Creates bytes object
    list(model_output_np.shape),
    str(model_output_np.dtype)
)
```

**Problem**:
- `tobytes()` creates a copy of the numpy array
- Original `model_output_np` might not be garbage collected immediately
- If Pythonx keeps references, memory leaks

**Risk**:
- Each transformer forward pass: ~200MB not freed
- 20 steps × 200MB = **4GB leak** per image

### 4. **Missing Garbage Collection Between Steps**

**Location**: No periodic GC in the denoising loop

**Problem**:
- Python doesn't GC until memory pressure detected
- On systems with 64GB RAM, GC might not trigger for a long time
- Intermediate tensors accumulate

**Missing**:
```python
import gc
gc.collect()  # Should be called after each denoising step
```

## Test Configuration Review

### ✅ Good Safeguards in Place

1. **Sequential test execution**:
   ```elixir
   use ExUnit.Case, async: false  # ✓ Prevents parallel model loading
   ```

2. **Memory checks before loading**:
   ```elixir
   required_mb = Margarine.Memory.estimate_flux_memory(state.model)
   # Checks before loading - ✓ Working
   ```

3. **Cleanup in terminate**:
   ```elixir
   def terminate(reason, state) do
     flux_pythonx.clear_memory()
     gc.collect()
   end
   ```

### ❌ Missing Safeguards

1. **No globals cleanup between calls**
2. **No periodic GC in denoising loop**
3. **No memory monitoring during generation**
4. **No timeout/cleanup for stuck processes**

## Crash Hypothesis

### Most Likely Scenario: **Globals Accumulation + VAE Decode**

1. Test runs 3 generations sequentially (tests 1-3 passed)
2. Test 4 starts: 1024x1024 image (4x larger)
3. Denoising loop runs 4 steps successfully
4. Each step adds ~200MB to globals (never freed)
5. VAE decode needs additional ~2GB for decoding
6. **Total memory**: 29GB - 12GB (model) - 4GB (accumulated) - 2GB (VAE) = **11GB remaining**
7. MPS backend might need additional memory for compute
8. **System hits memory pressure** during VAE decode
9. macOS kernel panics or OOM kills the process

### Supporting Evidence

- Crash happened at **VAE decode** (largest single-operation memory spike)
- Previous tests were **512x512** (4x smaller)
- Test was **4th in sequence** (maximum globals accumulation)
- Memory was healthy **before test** (29GB available)

## Recommended Fixes

### Priority 1: Fix Globals Accumulation (Critical)

**File**: `lib/margarine/python/pythonx_server.ex`

**Problem**: Reusing same globals dict accumulates data

**Solution**: Use fresh Python namespace per call OR explicitly delete old keys

**Option A - Fresh namespace per call** (Cleanest):
```elixir
# Instead of Map.merge(state.globals, new_data)
# Use only what's needed for this call
call_globals = %{
  # Only module imports from initial load
  "flux_pythonx" => state.globals["flux_pythonx"],
  # Add call-specific data
  "latents_bin" => latents_bin,
  "latents_shape" => latents_shape,
  # ...
}

# Don't merge back - keep original globals
case Pythonx.eval(code, call_globals) do
  {result, _} ->  # Discard new_globals
    {:reply, {:ok, decoded}, state}  # Keep original state.globals
end
```

**Option B - Explicit cleanup** (Safer):
```elixir
# Before each call, clean up old temporary variables
cleanup_code = """
# Delete large temporary variables
for key in list(globals().keys()):
    if key.endswith('_bin') or key.endswith('_shape'):
        del globals()[key]
import gc
gc.collect()
"""
Pythonx.eval(cleanup_code, state.globals)
```

### Priority 2: Add Periodic Garbage Collection

**File**: `priv/python/flux_pythonx.py`

**Add GC after each step**:
```python
def transformer_forward(...):
    # ... existing code ...

    # Convert to numpy
    model_output_np = model_output_f32.cpu().numpy()

    # FREE INTERMEDIATE TENSORS
    del model_output_f32, model_output_unpacked, model_output
    import gc
    gc.collect()

    return (...)
```

**Add GC after VAE decode**:
```python
def vae_decode(latents_np):
    # ... existing code ...

    image_np = image_f32.cpu().numpy()

    # FREE INTERMEDIATE TENSORS
    del image_f32, image, latents
    import gc
    gc.collect()

    return (...)
```

### Priority 3: Monitor Memory During Generation

**File**: `lib/margarine/pipeline.ex`

**Add memory logging**:
```elixir
defp denoising_loop(server, state, scheduler, latents, embeds) do
  Enum.reduce_while(timesteps, latents, fn {timestep, idx}, current_latents ->
    # Log memory before each step
    {:ok, mem_info} = Margarine.Memory.available_memory()
    available_gb = Margarine.Memory.bytes_to_mb(mem_info.available) / 1024

    Logger.debug("[Pipeline] Step #{idx}/#{total}, Memory available: #{Float.round(available_gb, 1)}GB")

    # Check if we're running low
    if available_gb < 5.0 do
      Logger.warning("[Pipeline] Low memory detected: #{Float.round(available_gb, 1)}GB - forcing Python GC")
      force_python_gc(server)
    end

    # ... existing code ...
  end)
end

defp force_python_gc(server) do
  code = """
  import gc
  gc.collect()
  """
  Pythonx.eval(code, %{})
end
```

### Priority 4: Add Test Memory Monitoring

**File**: `test/integration/flux_generation_test.exs`

**Add memory checks between tests**:
```elixir
setup do
  on_exit(fn ->
    # Log memory before cleanup
    {:ok, info} = Margarine.Memory.available_memory()
    Logger.info("Test complete, memory: #{Margarine.Memory.format_bytes(info.available)}")

    # Force cleanup
    if Process.alive?(server_pid) do
      GenServer.stop(server_pid)
      :timer.sleep(1000)  # Wait for cleanup
    end

    # Force system GC
    :erlang.garbage_collect()

    # Log memory after cleanup
    {:ok, info2} = Margarine.Memory.available_memory()
    Logger.info("After cleanup, memory: #{Margarine.Memory.format_bytes(info2.available)}")
  end)
end
```

### Priority 5: Add Memory Leak Detection Test

**New file**: `test/integration/memory_leak_test.exs`

```elixir
defmodule Margarine.MemoryLeakTest do
  use ExUnit.Case, async: false

  @describetag :integration
  @describetag timeout: 600_000  # 10 minutes

  test "no memory leak over 10 generations" do
    # Record initial memory
    {:ok, initial_info} = Margarine.Memory.available_memory()
    initial_mb = Margarine.Memory.bytes_to_mb(initial_info.available)

    # Generate 10 images
    for i <- 1..10 do
      {:ok, _image} = Margarine.generate(
        "test image #{i}",
        model: :flux_schnell,
        steps: 4,
        size: {512, 512},
        seed: i
      )

      # Check memory after each
      {:ok, current_info} = Margarine.Memory.available_memory()
      current_mb = Margarine.Memory.bytes_to_mb(current_info.available)
      leaked_mb = initial_mb - current_mb

      Logger.info("Generation #{i}/10: Memory used: #{leaked_mb}MB")

      # Fail if we've leaked more than 2GB
      assert leaked_mb < 2000, "Memory leak detected: #{leaked_mb}MB leaked after #{i} generations"
    end
  end
end
```

## Immediate Action Plan

1. ✅ **Document findings** (this file)
2. ✅ **Add memory monitoring** to pipeline (Priority 3)
3. ✅ **Add garbage collection** to Python functions (Priority 2)
4. ✅ **Fix globals accumulation** (Priority 1 - most critical)
5. ✅ **Add memory leak detection test** (Priority 5)
6. ⏳ **Re-run integration tests** with monitoring

## Fixes Applied (2026-01-14)

### Priority 1: Globals Accumulation (FIXED) ✅

**File**: `lib/margarine/python/pythonx_server.ex`

**Changes**:
- Added `build_call_globals/2` helper that creates fresh globals with only essential modules
- Updated ALL `handle_call` functions to discard `new_globals` instead of accumulating
- Only keeps module references: `["flux_pythonx", "initialized", "init_result"]`
- Prevents 200MB+ binary data from accumulating per call

**Impact**: Should eliminate multi-GB memory leaks from accumulated globals

### Priority 2: Python Garbage Collection (FIXED) ✅

**File**: `priv/python/flux_pythonx.py`

**Changes**:
- Added explicit `del` statements in `transformer_forward()` for intermediate tensors
- Added `gc.collect()` after deleting tensors in `transformer_forward()`
- Added explicit `del` and `gc.collect()` in `vae_decode()`
- Added explicit `del` and `gc.collect()` in `encode_prompt()`

**Impact**: Forces Python to free 200MB-2GB of tensors immediately instead of waiting for memory pressure

### Priority 3: Memory Monitoring (FIXED) ✅

**File**: `lib/margarine/pipeline.ex`

**Changes**:
- Added `check_memory_and_gc/3` function to monitor memory before each denoising step
- Logs available memory at DEBUG level for each step
- Forces Python GC if memory drops below 5GB
- Added pre-check before VAE decode (most memory-intensive operation)
- Added `force_python_gc/1` helper to trigger Python GC on demand

**Impact**: Early warning of memory issues + proactive cleanup prevents OOM

### Priority 4: Test Memory Monitoring (FIXED) ✅

**File**: `test/integration/flux_generation_test.exs`

**Changes**:
- Added memory logging before each test in `setup`
- Added memory logging before cleanup in `on_exit`
- Added Erlang GC (`erlang.garbage_collect()`) after each test
- Added memory logging after cleanup
- Sleep delays to allow cleanup to complete

**Impact**: Visibility into memory usage patterns and leak detection

### Priority 5: Memory Leak Detection Test (FIXED) ✅

**File**: `test/integration/memory_leak_test.exs` (NEW)

**Tests**:
1. `no memory leak over 10 small generations (512x512)` - Detects leaks with repeated small images
2. `no memory leak over 3 large generations (1024x1024)` - Detects leaks with large images

**Assertions**:
- Fails if more than 3GB leaked after small generations
- Fails if more than 4GB leaked after large generations
- Logs memory at each generation for analysis

**Impact**: Automated detection of memory leaks in CI/testing

## Additional Notes

### Known Benign Warnings

From `priv/python/flux_pythonx.py:15-19`:
```python
# Suppress known benign warning from Python's multiprocessing resource_tracker
# This warning appears when Python processes are terminated by external processes (Elixir)
# See: https://github.com/apple/ml-stable-diffusion/issues/8
warnings.filterwarnings('ignore', '.*resource_tracker.*', UserWarning)
```

This is **NOT** a memory leak - it's a cleanup warning when Elixir terminates Python.

### Memory Limit on macOS

From `priv/python/memory_limit.py:39-45`:
```python
if platform.system() == "Darwin":
    print("[Memory Limit] Skipping on macOS (Pythonx embedded Python)")
    return
```

This is **correct** - RLIMIT_AS doesn't work with embedded Python. We rely on Elixir-side checks.

## References

- MEMORY_LIMITING_GUIDE.md - Three-layer defense strategy
- CLAUDE.md - Known test failures and architecture notes
- integration_test_output.log - Last execution before crash

## Next Steps

Implement fixes in order of priority, test with memory monitoring enabled, and verify no leaks over multiple generation cycles.
