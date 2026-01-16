# Future Optimization Opportunities

## 1. Reduce Memory Copies with Pythonx References (High Impact)

**Source:** Conversation with Paulo Valente (Pythonx maintainer), 2026-01-15

### Current Implementation (Lots of Copying)

Every denoising step currently does 4 memory copies:

```elixir
# Elixir → Python (Step N)
latents_bin = Nx.to_binary(latents)           # COPY 1: Nx tensor → binary
# Python: np.frombuffer(latents_bin)          # COPY 2: binary → numpy array

# Python → Elixir (Step N)
result_bin = model_output_np.tobytes()        # COPY 3: numpy → binary
result = Nx.from_binary(result_bin)           # COPY 4: binary → Nx tensor
```

**Memory Cost for 1600×1600 image:**
- Each latent: ~200MB
- 4 denoising steps × 4 copies = **3.2GB of copying per image**
- Not counting prompt embeddings, pooled embeddings, etc.

### Proposed Optimization (Keep References)

Paulo Valente's suggestion:
> "Each time you do `to_binary` I think you're creating a new copy of the memory from the Nx tensor. If it's the same mutable data section that you wanna 'view' as an Nx tensor, I'd keep the original pythonx reference to reuse the same numpy array."

**Optimized approach:**

```elixir
defmodule Margarine.Pipeline.Optimized do
  # Generate latents, keep as Pythonx reference
  {:ok, latents_ref} = PythonxServer.generate_latents_as_ref(...)
  # Returns: #Pythonx.Object<numpy array>

  # Denoising loop - pass references around
  final_latents_ref = Enum.reduce(timesteps, latents_ref, fn {timestep, idx}, current_ref ->
    # Pass Pythonx reference directly (NO COPY!)
    {:ok, model_output_ref} = PythonxServer.transformer_forward_ref(
      current_ref,  # ← Pythonx reference, not binary!
      timestep,
      embeds.prompt_embeds_ref,
      embeds.pooled_embeds_ref,
      guidance_scale: state.guidance_scale
    )

    # Only convert to Nx when we MUST do scheduler math
    model_output = pythonx_ref_to_nx(model_output_ref)
    current_latents = pythonx_ref_to_nx(current_ref)

    # Scheduler step (pure Nx math)
    next_latents = FluxEuler.step(scheduler, model_output, idx, current_latents)

    # Convert back to Pythonx ref for next iteration
    nx_to_pythonx_ref(next_latents)
  end)

  # Final decode (converts ref → binary → image)
  {:ok, image} = PythonxServer.vae_decode_ref(final_latents_ref)
end

# New helper functions needed:
def pythonx_ref_to_nx(pythonx_obj) do
  # Get numpy array metadata from Python
  {:ok, shape, dtype, data_ptr} = get_numpy_info(pythonx_obj)
  # Create Nx tensor view (no copy!)
  Nx.from_pointer(data_ptr, shape, dtype)
end

def nx_to_pythonx_ref(nx_tensor) do
  # Create numpy array wrapping Nx memory (no copy!)
  shape = Nx.shape(nx_tensor)
  dtype = Nx.type(nx_tensor)
  data_ptr = Nx.to_pointer(nx_tensor)
  create_numpy_view(shape, dtype, data_ptr)
end
```

**Challenges:**
1. ❓ **Pythonx API support** - Need to verify Pythonx supports pointer-based operations
2. ❓ **Memory ownership** - Who owns the memory? BEAM or Python?
3. ❓ **GC coordination** - Ensure neither GC frees memory while other side uses it
4. ⚠️ **Complexity** - More complex than current binary passing
5. ⚠️ **Scheduler constraint** - Still need conversions for scheduler operations

**Potential Memory Savings:**
- Current: 4 copies × 200MB × 4 steps = **3.2GB**
- Optimized: 2 copies × 200MB × 4 steps = **1.6GB** (only for scheduler operations)
- **Savings: ~1.6GB (50% reduction in copy overhead)**

**Performance Impact:**
- Less memory pressure → less GC
- Fewer copies → faster execution
- Could enable larger images (2048×2048) on 64GB systems

### Implementation Checklist (When Ready)

1. **Research Phase:**
   - [ ] Verify Pythonx supports zero-copy pointer operations
   - [ ] Check if `Nx.from_pointer()` / `Nx.to_pointer()` exist
   - [ ] Understand memory ownership model (BEAM vs Python)
   - [ ] Test small prototype with single tensor

2. **Design Phase:**
   - [ ] Design API for `*_as_ref()` functions
   - [ ] Plan GC coordination strategy
   - [ ] Define error handling for memory ownership issues
   - [ ] Document memory lifetime contracts

3. **Implementation Phase:**
   - [ ] Add `generate_latents_as_ref/3`
   - [ ] Add `transformer_forward_ref/5`
   - [ ] Add `vae_decode_ref/1`
   - [ ] Add `pythonx_ref_to_nx/1` helper
   - [ ] Add `nx_to_pythonx_ref/1` helper
   - [ ] Update Pipeline to use reference-based API

4. **Testing Phase:**
   - [ ] Unit tests for reference conversions
   - [ ] Memory leak tests (ensure refs are cleaned up)
   - [ ] Integration tests (verify correctness)
   - [ ] Benchmark memory usage vs current implementation
   - [ ] Benchmark execution time vs current implementation

5. **Validation Phase:**
   - [ ] Compare image quality (ensure bit-exact match)
   - [ ] Run with memory profiler
   - [ ] Test with various image sizes
   - [ ] Verify GC behavior under load

### When to Implement

**Implement if:**
- ✅ Expanding to 2048×2048 and need every GB of savings
- ✅ Doing batch processing (100+ images) and seeing memory pressure
- ✅ Profiling shows conversion overhead is >10% of execution time
- ✅ Pythonx confirms zero-copy API is supported and stable

**Don't implement if:**
- ❌ Current approach works fine for your use case
- ❌ Only generating occasional images
- ❌ Pythonx doesn't support required pointer operations
- ❌ Complexity outweighs benefits

### Current Status

**Status:** 📋 Documented, not implemented
**Priority:** Low (current implementation is working)
**Blocker:** Need to verify Pythonx API capabilities
**Decision:** Measure performance before optimizing

### References

- Paulo Valente's advice: https://elixir-lang.slack.com/archives/C03EPRA3B/p1737077329292639
- Pythonx docs: https://hexdocs.pm/pythonx/
- Current implementation: `lib/margarine/python/pythonx_server.ex`

---

## 2. Optimize Scheduler Operations (Medium Impact)

### Opportunity

FluxEuler scheduler operations are currently creating intermediate tensors that could be reused:

```elixir
# Current (creates new tensors each step)
def step(scheduler, model_output, timestep_idx, latents) do
  timestep = Nx.to_number(scheduler.timesteps[timestep_idx])
  next_timestep = if timestep_idx < num_steps - 1, do: Nx.to_number(scheduler.timesteps[timestep_idx + 1]), else: 0.0

  # Each operation creates new tensor
  dt = Nx.subtract(next_timestep, timestep)
  scaled_output = Nx.multiply(model_output, dt)
  next_latents = Nx.add(latents, scaled_output)

  next_latents
end
```

**Potential optimization:** Pre-allocate output tensors and reuse them.

**Expected savings:** Minimal (scheduler is much smaller than model operations)
**Priority:** Very Low
**Status:** 📋 Documented for completeness

---

## 3. Batch Processing (High Impact for Production)

### Opportunity

Process multiple images in parallel with shared model instance:

```elixir
# Current: Sequential
images = Enum.map(prompts, fn prompt ->
  Margarine.generate(prompt, model: :flux_schnell)
end)

# Optimized: Batch with shared model
images = Margarine.generate_batch(prompts,
  model: :flux_schnell,
  batch_size: 4  # Process 4 at once
)
```

**Benefits:**
- Amortize model loading cost across multiple images
- Better GPU utilization
- Potentially 2-4x throughput

**Challenges:**
- Need batch support in Python layer
- Memory requirements scale with batch size
- More complex error handling

**Status:** 📋 Documented for future
**Priority:** Medium (production workload dependent)

---

## Notes

- This document captures optimization ideas that are **not currently needed**
- Current implementation is production-ready and working well
- Implement these only when you have **measured evidence** they're needed
- **Premature optimization is the root of all evil** - Donald Knuth

**Last Updated:** 2026-01-15
