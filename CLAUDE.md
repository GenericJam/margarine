# Margarine - Elixir Image Generation Library 🧈

**"I Can't Believe It's Not Real Art!"**

> **Why "Margarine"?** Image generation is artificial - it emulates real photographs and artists. Like margarine vs butter, it's synthetic but pretty good! A playful reminder not to take ourselves too seriously. 😄

---

## Development Tools

### Semantic Code Search with C-K (seek)

We have `ck` (C-K, pronounced "seek") available for semantic grep - a way to cast a wider net when searching through code. When searching for patterns, consider using `ck` for semantic understanding beyond simple text matching.

---

## 🚧 CONTINUATION PLAN: Stable Diffusion XL Integration (Phase 2)

**Status:** FLUX integration complete! Now adding SDXL for text2img and img2img support.

## 🚧 CONTINUATION PLAN: FLUX Integration (Phase 1.5) ✅ COMPLETE

**Status:** Phase 1 MVP complete (10/10 beads) - all infrastructure ready. FLUX generation working!

**What's Done:**
- ✅ All core modules (Config, Memory, Image, Pipeline, Schedulers)
- ✅ Public API structure (Margarine.generate/1 and /2)
- ✅ Pure Nx FluxEuler scheduler
- ✅ Pythonx integration layer foundations
- ✅ 119 tests passing (75.3% coverage)
- ✅ Complete documentation and examples
- ✅ Hex package ready
- ✅ Python file copied: `priv/python/flux_pythonx.py` from working `imagine` project

**What's Missing:** The actual Python FLUX model integration (currently returns stub error)

**Remaining Beads (in order):**

1. **margarine-z7f**: Create `Margarine.Python.PythonxServer` GenServer
   - Copy/adapt from `~/code/imagine/lib/imagine/inference/pythonx_server.ex`
   - Key functions: `initialize_model/3`, `encode_prompt/3`, `transformer_forward/6`, `vae_decode/2`
   - Use Pythonx for zero-copy tensor sharing
   - Handle FLUX model loading and inference

2. **margarine-5se**: Wire `Pipeline.generate/1` to call PythonxServer
   - Add `generate/1` function to Pipeline module
   - Flow: prepare → encode_prompt → denoising_loop → vae_decode
   - Use FluxEuler scheduler for timesteps
   - Return final image tensor

3. **margarine-bkz**: Update `Margarine.generate/2` to use real implementation
   - Remove stub "not yet implemented" error
   - Call `Pipeline.generate/1` with prepared state
   - Return actual generated image

4. **margarine-0g8**: Test end-to-end with example scripts
   - Run `elixir examples/basic.exs` - should generate image
   - Run `elixir examples/advanced.exs` - should generate 3 images
   - Verify images are saved as PNG files

5. **margarine-15r**: Run integration tests with real FLUX
   - `mix test --only integration` should pass
   - Tests in `test/integration/flux_generation_test.exs`
   - Verify reproducibility, seeds, and error handling

**Reference Implementation:**
- Working code in `~/code/imagine`
- See `~/code/imagine/CLAUDE.md` for architecture details
- Key files:
  - `lib/imagine/inference/pythonx_server.ex` - GenServer
  - `lib/imagine/pipelines/flux_text2image.ex` - Pipeline
  - `priv/python/flux_pythonx.py` - Python (already copied!)

**Testing Strategy:**
- Start with basic.exs to verify generation works
- Then run integration tests
- Keep TDD practices but don't over-test during integration
- Focus on getting it working first, then add tests if coverage drops

**Estimated Effort:** 2-3 hours for complete integration

---

## 📋 SDXL Integration Plan (Phase 2)

### Reference Implementation

Working SDXL code in `~/code/genericjam`:
- **Elixir modules**: `lib/genericjam/image_generation/service.ex`
- **Python inference**: `priv/python/model_inference.py`
- **Architecture**: Port-based (not Pythonx) with JSON communication
- **Features**: text2img + img2img with `denoising_strength` parameter

### Implementation Approach

**Strategy**: Adapt FLUX Pythonx architecture for SDXL while reusing as much infrastructure as possible.

**Architecture (Same as FLUX):**
```
Elixir Pipeline (orchestration, scheduling, state management)
    ↓
PythonxServer GenServer (holds loaded model, shared memory)
    ↓
Python (model inference only via Pythonx)
    ↓
Nx ←→ Shared Memory ←→ NumPy (zero-copy)
```

**Pipeline Unification:**
- Text2img: Generate random noise → pass to unified pipeline
- Img2img: VAE encode image → pass to unified pipeline
- **After first step, both are identical** - just denoising latents!
- Single `Pipeline.denoise/1` function handles both cases

**Design Principle: Universal Composability** 🎨
A key goal is to **mix and match schedulers AND models** during generation.

**Core Insight**: Everything is img2img + prompt. Text2img is just img2img starting from pure noise.

**At each step, you have:**
- Current latent (noisy image at some noise level)
- Current timestep (how noisy it is: 1.0 = pure noise, 0.0 = clean image)
- Prompt (what you want)

**You can hand this to ANY model** and say "denoise this one step" because:
- Both FLUX and SDXL use the same VAE (same latent space)
- Both use 16-channel latents at 1/8 resolution
- A "partially denoised image" is just an image - any model can continue from it
- The timestep tells the model "treat this as if it's at step X"

**Two types of composability:**

1. **Scheduler Mixing** (same model, different scheduler)
   - SDXL + DDIM (10 steps) → SDXL + Euler (10 steps)
   - Same model, different denoising strategy

2. **Cross-Model Mixing** (different models, any schedulers)
   - FLUX + Euler (5 steps) → SDXL + DDIM (10 steps) → FLUX + Euler (5 steps)
   - Each model continues denoising from where the last left off
   - Models bring their own "style" to the denoising process

**Why this works:**
- Shared latent space (same VAE)
- Denoising is a continuous process
- Each step just says "make it slightly less noisy toward the prompt"
- Different models = different artistic interpretations of the same denoising task

**Benefits:**
- Mix FLUX's artistic style with SDXL's detail refinement
- Experiment with model transitions at different noise levels
- Show step-by-step progression (not a black box!)
- Elixir schedulers give us full control

**Key differences from FLUX**:
1. **Dual text encoders**: SDXL uses CLIP + CLIP-with-projection (not T5)
2. **UNet instead of Transformer**: Different model architecture
3. **Scheduler options**: Can use DDIM, Euler, DPMSolver++ (not just EulerFlow)
4. **Time IDs**: SDXL requires additional time/size conditioning
5. **IMG2IMG**: Requires VAE encoder to convert init image to latents
6. **Latent compatibility**: Same 16-channel latent space as FLUX enables model mixing

### Beads (Phase 2.1: Text2Image)

1. **margarine-sdxl-py** - Create SDXL Python module
   - Copy `priv/python/flux_pythonx.py` → `priv/python/sdxl_pythonx.py`
   - Adapt from `genericjam/priv/python/model_inference.py`
   - Functions: `initialize_model`, `encode_prompt`, `unet_forward_cfg`, `vae_decode`
   - Handle dual CLIP encoders + pooled embeddings
   - Add `get_time_ids()` helper for SDXL conditioning

2. **margarine-sdxl-config** - Add SDXL model config
   - Update `lib/margarine/config.ex`
   - Add `:sdxl_base` and `:sdxl_turbo` model configs
   - Model IDs: `stabilityai/stable-diffusion-xl-base-1.0`, `stabilityai/sdxl-turbo`
   - Defaults: 20 steps (base), 1 step (turbo), guidance 7.5

3. **margarine-sdxl-scheduler** - Add DDIM scheduler
   - Create `lib/margarine/schedulers/ddim.ex`
   - Pure Nx implementation of DDIM algorithm
   - Port from `genericjam/lib/genericjam/diffusion/schedulers/ddim.ex`
   - Support both SDXL and FLUX (scheduler-agnostic)

4. **margarine-sdxl-server** - SDXL PythonxServer
   - Create `lib/margarine/python/sdxl_pythonx_server.ex`
   - Similar to FluxPythonxServer but call `sdxl_pythonx.py`
   - Functions: `encode_prompt`, `unet_forward`, `vae_decode`, `generate_latents`
   - Memory checking (SDXL base ~7GB, turbo ~7GB)

5. **margarine-sdxl-pipeline** - SDXL generation pipeline
   - Create `lib/margarine/pipelines/sdxl_text2image.ex`
   - Modeled after `lib/margarine/pipeline.ex` (FLUX version)
   - Steps: prepare → encode_prompt → scheduler_init → initial_latents → denoising → vae_decode
   - Support both DDIM and Euler schedulers

6. **margarine-api-sdxl** - Update public API
   - Update `lib/margarine.ex` to route SDXL models
   - Detect model type (flux_* vs sdxl_*) and use appropriate pipeline
   - Keep same `Margarine.generate/2` interface

### Beads (Phase 2.2: Image-to-Image)

7. **margarine-img2img-py** - Add VAE encoder to Python
   - Add `vae_encode(image_np)` to both `flux_pythonx.py` and `sdxl_pythonx.py`
   - Converts image [H,W,3] uint8 → latents [B,C,H//8,W//8] float32
   - Reference: `genericjam/priv/python/model_inference.py` `vae_encode()`

8. **margarine-img2img-image** - Image preprocessing
   - Update `lib/margarine/image.ex`
   - Add `prepare_init_image/2` - resize and normalize to [-1, 1]
   - Add `to_latent_size/1` - calculate latent dimensions (H//8, W//8)

9. **margarine-img2img-scheduler** - Add noise to latents
   - Add `add_noise/3` to scheduler behavior
   - Implement in FluxEuler and DDIM
   - Formula: `noisy = sqrt(alpha) * latents + sqrt(1-alpha) * noise`
   - Used for img2img starting point

10. **margarine-unified-pipeline** - Unified denoising pipeline
    - **Key insight**: Text2img and img2img are the same after getting initial latents!
    - Text2img: `prepare_random_latents/2` → `denoise_loop/1`
    - Img2img: `prepare_image_latents/3` (encode + add noise) → `denoise_loop/1`
    - **Same `denoise_loop/1` for both** - just takes latents + starting timestep
    - Pipeline state: `%{latents, timestep, prompt_embeds, scheduler, steps_remaining}`
    - Example: Img2img with strength=0.7 starts at timestep 0.7, does 14/20 steps

11. **margarine-img2img-api** - Public IMG2IMG API
    - Update `lib/margarine.ex`
    - Add `Margarine.img2img/2` function
    - Params: `init_image`, `prompt`, `denoising_strength`, etc.
    - Validate init_image path exists

### Beads (Phase 2.3: Universal Composability) 🎨

**Both scheduler mixing AND cross-model mixing - it's all just img2img!**

12. **margarine-stepped-api** - Universal stepped generation API
    - Add `Margarine.init_generation/2` - Initialize with prompt + starting latents
    - Add `Margarine.step_generation/3` - Run N steps with model + scheduler + timestep
    - Add `Margarine.finalize_generation/1` - Decode latents to image
    - Add `Margarine.get_current_latents/1` - Inspect intermediate state
    - Session state: `%{latents, timestep, total_steps_taken}`
    - **Key**: Pass current timestep to model ("pretend you're at step X")

13. **margarine-session-manager** - Generation session storage
    - Create `lib/margarine/session_manager.ex` - ETS-based session storage
    - Store active generation sessions with TTL (30 min)
    - Functions: `create_session/1`, `get_session/1`, `update_session/2`, `delete_session/1`
    - Store prompt per-model (FLUX needs T5 embeds, SDXL needs CLIP)
    - Auto-cleanup expired sessions

14. **margarine-timestep-management** - Universal timestep handling
    - Normalize timesteps to [0.0, 1.0] range across all schedulers
    - Map to model-specific ranges (FLUX: 1.0→0.0, SDXL: depends on scheduler)
    - Track "noise level" independently of model/scheduler
    - Allow arbitrary starting points (for img2img and model switching)

15. **margarine-universal-api** - Composable pipeline API
    - Add `Margarine.generate_composed/2` - Universal composition
    - Accepts `pipeline:` with `{model, scheduler, steps}` tuples
    - Works for same-model (scheduler mixing) OR cross-model
    - Handles prompt encoding per-model automatically
    - Example: `[{:flux_schnell, :euler, 5}, {:sdxl_base, :ddim, 10}, {:flux_schnell, :euler, 5}]`

### Beads (Phase 2.5: Testing & Documentation)

17. **margarine-sdxl-tests** - SDXL integration tests
    - Add `test/integration/sdxl_generation_test.exs`
    - Test text2img with `sdxl_base`
    - Test reproducibility with seeds
    - Mark with `@tag :integration`

18. **margarine-img2img-tests** - IMG2IMG integration tests
    - Test img2img with both FLUX and SDXL
    - Test various denoising strengths (0.3, 0.5, 0.7)
    - Verify output matches init image dimensions

19. **margarine-composability-tests** - Universal composability tests
    - Test scheduler mixing: SDXL with DDIM (10) → Euler (10)
    - Test cross-model: FLUX (5) → SDXL (10) → FLUX (5)
    - Test different transition points (early, mid, late)
    - Verify latents remain valid across transitions
    - Compare composed vs single-model generation quality
    - Save intermediate images to show progression (not a black box!)

20. **margarine-sdxl-docs** - Documentation
    - Update README with SDXL examples
    - Document model differences (FLUX vs SDXL)
    - Add img2img usage examples
    - Add scheduler mixing examples
    - Add cross-model mixing examples
    - Document core insight: "everything is img2img + prompt"
    - Show step-by-step image progression
    - Update HexDocs

### File Structure

```
lib/margarine/
├── pipeline.ex                   # FLUX unified pipeline (existing)
├── pipelines/
│   └── sdxl.ex                   # SDXL unified pipeline (new, same structure as FLUX)
├── python/
│   ├── pythonx_server.ex         # FLUX PythonxServer (existing)
│   └── sdxl_pythonx_server.ex    # SDXL PythonxServer (new)
├── schedulers/
│   ├── flux_euler.ex             # Rectified flow (existing)
│   └── ddim.ex                   # DDIM scheduler (new)

priv/python/
├── flux_pythonx.py               # FLUX inference (existing)
└── sdxl_pythonx.py               # SDXL inference (new)

test/integration/
├── flux_generation_test.exs      # FLUX tests (existing)
├── sdxl_generation_test.exs      # SDXL tests (new)
└── img2img_test.exs              # IMG2IMG tests (new)
```

**Pipeline Architecture (All Elixir):**

```elixir
# Text2img flow
Margarine.generate(prompt, model: :sdxl_base)
  ↓
Pipeline.prepare_random_latents()  # Pure Elixir: Nx.random_normal()
  ↓
Pipeline.denoise_loop()            # Pure Elixir orchestration
  ↓ (each step)
  Scheduler.step()                 # Pure Nx math
  PythonxServer.unet_forward()     # Python inference (zero-copy)
  ↓
Pipeline.vae_decode()              # Python VAE (zero-copy)

# Img2img flow
Margarine.img2img(init_image, prompt, strength: 0.7)
  ↓
Image.load_and_preprocess()        # Pure Elixir: Vix/Image
  ↓
Pipeline.prepare_image_latents()   # Python VAE encode + Nx noise
  ↓
Pipeline.denoise_loop()            # 👈 SAME FUNCTION AS TEXT2IMG!
  (continues from timestep 0.7)
```

**Key Design:**
- **All orchestration in Elixir** (scheduling, state, control flow)
- **Python only for inference** (UNet forward, VAE encode/decode)
- **GenServer holds model** between steps (no reloading)
- **Shared memory** (Nx ↔ NumPy) for zero-copy transfers

**Current FLUX Pipeline (Reference):**
Look at `lib/margarine/pipeline.ex` - this is the pattern to follow for SDXL:

```elixir
defmodule Margarine.Pipeline do
  # Main entry point
  def generate(state) do
    with {:ok, server} <- get_or_start_server(state.model),
         {:ok, embeds} <- encode_prompt(server, state),
         {:ok, scheduler} <- initialize_scheduler(state),
         {:ok, latents} <- generate_initial_latents(server, state),
         {:ok, final_latents} <- denoising_loop(server, state, scheduler, latents, embeds),
         {:ok, image} <- decode_image(server, final_latents) do
      {:ok, image}
    end
  end

  # Pure Elixir: Generate random noise
  defp generate_initial_latents(server, state) do
    PythonxServer.generate_latents(server, height, width, seed)
  end

  # Pure Elixir orchestration
  defp denoising_loop(server, state, scheduler, latents, embeds) do
    Enum.reduce_while(timesteps, latents, fn {timestep, idx}, current_latents ->
      # Pure Nx scheduler math
      # Python inference (zero-copy)
      # Update latents
    end)
  end
end
```

**SDXL will be identical structure**, just:
- Different model server (SDXLPythonxServer)
- Different embeddings (CLIP instead of T5)
- Same denoising loop pattern

### API Examples

**Text2Image with SDXL**:
```elixir
# SDXL Base (20 steps, high quality)
{:ok, image} = Margarine.generate("a red panda", model: :sdxl_base, steps: 20)

# SDXL Turbo (1 step, fast)
{:ok, image} = Margarine.generate("a red panda", model: :sdxl_turbo, steps: 1)
```

**Image-to-Image**:
```elixir
# Light modification (keep most of original)
{:ok, image} = Margarine.img2img(
  init_image: "photo.png",
  prompt: "turn into a watercolor painting",
  denoising_strength: 0.3,  # 30% change
  model: :sdxl_base
)

# Heavy modification
{:ok, image} = Margarine.img2img(
  init_image: "sketch.png",
  prompt: "realistic photograph",
  denoising_strength: 0.8,  # 80% change
  model: :flux_schnell
)
```

**Universal Composability** 🎨:
```elixir
# Initialize generation with FLUX
{:ok, session_id} = Margarine.init_generation(
  prompt: "a cyberpunk cityscape at sunset",
  model: :flux_schnell,
  size: {1024, 1024},
  seed: 42
)

# Do first 5 steps with FLUX
{:ok, session_id} = Margarine.step_generation(
  session_id,
  model: :flux_schnell,
  steps: 5,
  scheduler: :euler
)

# Continue 10 steps with SDXL for detail refinement
{:ok, session_id} = Margarine.step_generation(
  session_id,
  model: :sdxl_base,
  steps: 10,
  scheduler: :ddim
)

# Finish last 5 steps with FLUX for artistic style
{:ok, session_id} = Margarine.step_generation(
  session_id,
  model: :flux_schnell,
  steps: 5,
  scheduler: :euler
)

# Decode final latents to image
{:ok, image} = Margarine.finalize_generation(session_id)

# Bonus: Get intermediate images to show progression!
{:ok, intermediate} = Margarine.get_current_latents(session_id)
                      |> Margarine.decode_latents(:flux_schnell)
```

**Simplified Composable API**:
```elixir
# Cross-model mixing in one call
{:ok, image} = Margarine.generate_composed(
  prompt: "a cyberpunk cityscape at sunset",
  size: {1024, 1024},
  seed: 42,
  pipeline: [
    {:flux_schnell, :euler, 5},   # FLUX's artistic style
    {:sdxl_base, :ddim, 10},      # SDXL's detail refinement
    {:flux_schnell, :euler, 5}    # FLUX final polish
  ],
  save_intermediates: true  # Save images at each transition
)

# Same-model scheduler mixing (also works!)
{:ok, image} = Margarine.generate_composed(
  prompt: "mountain landscape",
  model: :sdxl_base,
  pipeline: [
    {:sdxl_base, :ddim, 10},   # Exploration phase
    {:sdxl_base, :euler, 10}   # Convergence phase
  ]
)
```

### Key Technical Details

**SDXL Text Encoding**:
- Dual encoders: CLIP-ViT-L (768d) + CLIP-ViT-G-with-projection (1280d)
- Concatenated: 2048d final embedding
- Pooled embeddings from second encoder used for conditioning

**Time IDs (SDXL-specific)**:
```python
# Additional conditioning beyond text
time_ids = [
  original_height,   # 1024
  original_width,    # 1024
  crop_top,          # 0
  crop_left,         # 0
  target_height,     # 1024
  target_width       # 1024
]
```

**IMG2IMG Denoising Strength**:
- `strength = 0.0`: No change (start at step 0)
- `strength = 0.5`: Moderate change (start at step 10/20)
- `strength = 1.0`: Complete change (start at step 20/20, equivalent to text2img)
- Formula: `start_step = int((1 - strength) * num_steps)`

**Memory Requirements**:
- SDXL Base: ~7GB VRAM (vs FLUX's ~14GB)
- SDXL Turbo: ~7GB VRAM
- Lower than FLUX, so should work well on 64GB unified memory

**The Core Insight: Everything is IMG2IMG** 🎨

Text2img is just a special case of img2img where you start with pure noise!

```
At any point in generation, you have:
1. Latent (partially denoised image)
2. Timestep (noise level: 1.0 = pure noise, 0.0 = clean)
3. Prompt (what you want)

You can hand these to ANY model and say "denoise one more step"
```

**Universal Latent Compatibility**:
Both FLUX and SDXL use the same VAE, so latents are interchangeable:

```
Shared Latent Space:
- Channels: 16 (both models)
- Spatial resolution: H/8 × W/8 (both use 8x downsampling)
- Value range: Normalized by VAE scaling factor (~0.13)
- Data type: float32 for Nx operations
```

**How Model Switching Works**:
1. **FLUX denoises for 5 steps** → produces latent + timestep
2. **Hand to SDXL** → "here's a latent at timestep 0.75, denoise it"
3. **SDXL denoises for 10 steps** → produces latent + new timestep
4. **Hand back to FLUX** → "here's a latent at timestep 0.25, finish it"

Each model brings its own "artistic interpretation" to the denoising process.

**What to handle when switching**:
1. **Prompt embeddings**: Re-encode prompt with target model's encoders
   - FLUX: T5-XXL (4096d)
   - SDXL: CLIP concat (2048d) + pooled embeds + time_ids
2. **Timestep mapping**: Normalize to [0,1], map to model-specific convention
3. **That's it!** Latents just flow through unchanged

**Why show progression**:
- Not a black box - see how image evolves at each transition
- Understand what each model contributes
- Experiment with transition timing (early vs late)
- Educational and debuggable

### Testing Strategy

**Fast tests (unit)**:
- Test schedulers (DDIM) with mock data
- Test image preprocessing functions
- Test parameter validation

**Integration tests**:
- Text2img: 512x512 for speed, 4 steps
- IMG2IMG: Use test fixtures, small images
- Test both SDXL and FLUX pipelines
- Sequential execution (async: false)

### Estimated Effort

**Phase 2.1 (Text2Image)**: ~4-6 hours
- Python module adaptation
- DDIM scheduler implementation
- Pipeline & server setup

**Phase 2.2 (IMG2IMG)**: ~3-4 hours
- VAE encoder integration
- Scheduler noise addition
- IMG2IMG pipeline

**Phase 2.3 (Universal Composability)**: ~4-6 hours 🎨
- Stepped generation API (simpler than expected!)
- Session management (ETS)
- Timestep normalization and tracking
- Per-model prompt encoding
- Works for scheduler mixing AND cross-model (same architecture)

**Phase 2.5 (Testing/Docs)**: ~3-4 hours
- Integration tests (SDXL, IMG2IMG, composability)
- Document core insight: "everything is img2img"
- Show step-by-step image progression examples
- Example scripts

**Total**: ~14-18 hours for complete SDXL + IMG2IMG + Universal Composability

**Note**: Universal composability is simpler than originally thought! Once you understand "everything is img2img + timestep", both scheduler mixing and cross-model mixing use the same infrastructure.

---

## 🚨 CRITICAL: TEST-DRIVEN DEVELOPMENT (TDD) 🚨

**WE ARE PRACTICING TDD ON THIS PROJECT. NO EXCEPTIONS.**

### The Rules:

1. **Tests FIRST, code SECOND**
   - Write the test before writing implementation code
   - Run the test and watch it fail
   - Write minimal code to make it pass
   - Refactor if needed
   - Repeat

2. **80%+ Code Coverage Required**
   - Target: 80% minimum coverage
   - If coverage falls below 80%, provide written justification OR write more tests
   - Use `mix test --cover` to verify coverage
   - Coverage reports must be included in PR descriptions

3. **Test Quality Over Quantity**
   - Mocks are allowed but must be justified
   - At least one "real" integration test per major feature that actually generates an image
   - Fast tests (mocked/1-step) for unit testing and development speed
   - Slow tests (real generation) for integration/CI validation
   - Tests must verify behavior, not implementation details

4. **Image Generation Testing Strategy**

   **Fast Tests (majority, for development):**
   - Mock the Python layer entirely
   - Use 1-step generation for quick validation
   - Test parameter validation and error handling
   - Test tensor shape/size validation
   - Use tiny test images (64x64 or 128x128)

   **Real Tests (critical path validation):**
   - Actually load FLUX model and generate images
   - Tag with `@tag :integration` or `@tag :slow`
   - Verify image dimensions, format, non-blank output
   - Run these in CI but maybe not on every commit
   - Example: `mix test --only integration` for full validation

5. **When Tests Can Be Changed**

   **Valid reasons:**
   - Requirements changed (document why in commit message)
   - Test was testing implementation detail, not behavior
   - Discovered edge case that test didn't account for
   - Refactoring improved design and test needs updating

   **INVALID reasons:**
   - Test is failing and I don't want to fix the code
   - Code is "close enough" to what test expects
   - "It works on my machine"
   - Convoluted reasoning to justify changing test to match buggy code

6. **Coverage Exemptions**

   May have <80% coverage if:
   - Interfacing with external systems that can't be tested (document in CLAUDE.md)
   - Generated code (boilerplate, migrations)
   - Intentionally untested exploration code (must be marked with `# TODO: Add tests`)

   All exemptions must be explicitly documented and justified.

### Test Organization

```
test/
├── margarine_test.exs              # Public API tests
├── margarine/
│   ├── config_test.exs             # Configuration tests
│   ├── models/
│   │   └── flux_test.exs           # FLUX model tests (fast + slow)
│   ├── python/
│   │   ├── server_test.exs         # Python integration (mocked)
│   │   └── runtime_test.exs        # Process management
│   ├── pipeline_test.exs           # Pipeline coordination
│   ├── image_test.exs              # Image encoding/decoding
│   └── telemetry_test.exs          # Telemetry events
├── integration/
│   ├── flux_generation_test.exs    # @tag :integration - Real FLUX generation
│   └── end_to_end_test.exs         # @tag :integration - Full pipeline (future)
└── test_helper.exs                 # Test setup, shared fixtures
```

### Running Tests

**Fast tests (default):**
```bash
mix test              # Runs all tests except integration tests
mix test --cover      # With coverage report
```

**Integration tests (requires model download, 16GB+ RAM):**
```bash
mix test --only integration         # Only integration tests
mix test --include integration      # All tests including integration
```

**Integration tests are excluded by default** because they:
- Download ~12GB FLUX model on first run
- Require 16GB+ RAM or 12GB+ VRAM
- Take 5-30 seconds per test
- **MUST run sequentially (async: false)** to avoid loading multiple model instances
- Each test has 5-minute timeout to handle model loading

**CRITICAL: Memory Management**
- Integration tests use `async: false` to prevent parallel execution
- Tests share a single PythonxServer instance via `setup_all` callback
- This prevents loading the 12GB model multiple times simultaneously
- Running tests in parallel will cause OOM kills on machines with <32GB RAM

**Known Benign Warning:**
You may see this warning at the end of integration tests:
```
resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
```
This is a known benign warning from Python's multiprocessing module when the Python
process is terminated by Elixir/BEAM. It does NOT indicate a real memory leak.
The semaphore is properly cleaned up by the OS at process exit.
See: https://github.com/apple/ml-stable-diffusion/issues/8

For CI/CD, run fast tests on every commit, integration tests on merges to main.

### Elixir Code Style (Credo Standards)

**We follow Credo's default rules for consistent, idiomatic Elixir code.**

1. **Control Flow Preferences:**
   - **`case`** - Most native to BEAM, use for pattern matching on values
   - **`cond`** - Use for multiple conditions
   - **`if/else`** - Use when working with booleans (predicates ending in `?`)
   - **`with`** - Use for chaining operations that can fail
   - **NEVER `unless`** - Outlawed by Credo, use `if not` instead

2. **When to Use `if/else`:**
   - ✅ **GOOD**: When you already have a boolean
     ```elixir
     if current_user_present? do
       show_dashboard()
     else
       redirect_to_login()
     end
     ```
   - ❌ **BAD**: When pattern matching would be clearer
     ```elixir
     if value == {:ok, "foo"} do  # Should use case instead
       handle_foo()
     end
     ```

3. **Pattern Matching Examples:**
   ```elixir
   # GOOD: Use case for pattern matching
   case Nx.type(tensor) do
     {:u, 8} -> validate_shape(tensor)
     type -> {:error, "Invalid type: #{inspect(type)}"}
   end

   # BAD: Using if/else for pattern matching
   if Nx.type(tensor) == {:u, 8} do
     validate_shape(tensor)
   else
     {:error, "Invalid type"}
   end

   # NEVER: Using unless
   unless valid? do  # ❌ Don't do this
     {:error, "Invalid"}
   end
   ```

4. **Credo Integration:**
   - Run `mix credo` before committing
   - Fix all warnings
   - Accept Credo's default styling rules (community standard)

### Type Safety with Dialyzer

**All functions must have type specs!**

1. **Type Specs Required**
   - Add `@spec` for all public functions
   - Add `@type` for custom types
   - Use proper Elixir typespecs (not just `any()`)
   - Document complex types

2. **Dialyzer Checks**
   - Run `mix dialyzer` periodically during development
   - Add dialyxir to dev dependencies
   - Fix all dialyzer warnings before merging
   - No excuses for "dialyzer doesn't understand this"

3. **Example**
   ```elixir
   @type generation_opts :: [
     model: atom(),
     steps: pos_integer(),
     guidance_scale: float(),
     seed: non_neg_integer() | nil,
     size: {pos_integer(), pos_integer()}
   ]

   @spec generate(String.t(), generation_opts()) :: {:ok, Nx.Tensor.t()} | {:error, String.t()}
   def generate(prompt, opts \\ []) do
     # ...
   end
   ```

### Elixir Version Support

**Development and Testing Strategy:**

1. **Primary Development: Elixir 1.19**
   - Use `mise use elixir@1.19` for daily development
   - Take advantage of latest features where appropriate
   - This is what we test most frequently

2. **Minimum Support: Elixir 1.14**
   - Specified in mix.exs as `elixir: "~> 1.14"`
   - Test compatibility periodically with `mise use elixir@1.14`
   - May adjust if 1.14 proves too restrictive
   - Don't use features introduced after 1.14

3. **CI Testing (future)**
   - Test on both 1.14 and 1.19 in CI
   - Ensures compatibility across range

4. **Switching Versions with Mise**
   ```bash
   # Development (default)
   mise use elixir@1.19

   # Compatibility testing
   mise use elixir@1.14
   mix deps.get
   mix test
   mix dialyzer

   # Switch back
   mise use elixir@1.19
   ```

### Believing in US

**YES, I BELIEVE IN US!** 🚀

We're going to write solid, well-tested code that:
- Works reliably
- Fails gracefully with clear errors
- Is maintainable by others
- Has proper type specs and passes Dialyzer
- Supports Elixir 1.14-1.19
- Makes us proud to show in job interviews
- Actually generates beautiful images

Let's ship quality code, not just code that "works for now."

---

## Project Vision

Margarine is an Elixir library that brings FLUX and Stable Diffusion image generation capabilities to the Elixir ecosystem using Nx, Pythonx, and direct PyTorch integration. The goal is to make AI image generation feel native to Elixir while leveraging the Python ML ecosystem under the hood.

## Why "Margarine"?

Because it's a butter substitute, just like this is a Python substitute for Elixir developers. Plus, it's fun and memorable.

## Local Reference Projects

**Important:** This is not starting from scratch! We have working implementations to reference:

1. **`~/code/imagine`** - FLUX implementation
   - Working FLUX integration with Pythonx
   - Zero-copy tensor transfer
   - MPS/Metal backend support
   - This is our primary reference for Phase 1

2. **`~/code/imagine_demo`** - FLUX demo/examples
   - Example usage patterns
   - UI integration patterns
   - Reference for documentation examples

3. **`~/code/genericjam`** - Stable Diffusion implementation
   - Original working SD implementation
   - Was migrated to `~/code/imagine` (needs verification)
   - Reference for Phase 4 when we backfill SD support

**Strategy:** Extract and refine the working code from these projects into a clean, production-ready library. Don't reinvent the wheel - we've already solved the hard problems!

## Core Design Principles

1. **Elixir-First API**: Users should feel like they're using an Elixir library, not calling Python
2. **Zero-Copy Performance**: Leverage Pythonx for efficient tensor transfer between Elixir and Python
3. **Zero Python Dependency Management**:
   - Use Pythonx (NOT Ports) for Python integration
   - Python dependencies managed via UV through Pythonx
   - Users should NEVER have to run `pip install` or manage virtualenvs
   - Dependencies should be automatically handled on first run
4. **Memory Safety**:
   - Check available system memory before loading models
   - Prevent OOM crashes that could take down the entire machine
   - Clear memory usage warnings and graceful failures
   - Model size estimates and minimum memory requirements documented
5. **Backend Flexibility**: Support multiple Nx backends following Arcana's pattern
   - EMLX (Apple Silicon/Metal)
   - EXLA (CUDA/ROCm/CPU via XLA)
   - Torchx (PyTorch backend via eager execution)
   - User configures backend in their own `config/config.exs`
   - Margarine remains agnostic to backend choice
6. **Streaming Results**: Return intermediate images during generation (denoising steps)
7. **Production Ready**: Proper error handling, telemetry, and documentation
8. **Modular Schedulers & Pipelines**: Following the `imagine` architecture pattern
   - Schedulers are generalized and composable
   - Mix and match schedulers with different algorithms
   - Clean separation between scheduling logic and model inference
   - Support for multiple scheduler types (Euler, DDIM, DPM++, etc.)

## Python Installation Strategy

**Approach: Automatic via Pythonx.uv_init() (Zero User Configuration)**

Margarine uses `Pythonx.uv_init()` called during `Application.start/2` to:
1. Automatically download and install Python (>=3.11) via UV
2. Create an isolated virtual environment for the project
3. Install all dependencies (torch, diffusers, transformers, etc.)
4. Cache everything for subsequent runs (instant startup after first run)

**First Run Experience:**
```
[Margarine] Initializing Python environment via UV...
[Margarine] Downloading Python 3.11... (~100MB, first run only)
[Margarine] Installing dependencies (torch, diffusers, etc.)... (~500MB)
[Margarine] This may take 2-5 minutes on first run...
[Margarine] ✓ Python environment ready!
```

**Subsequent Runs:**
Instant - everything is cached.

**Production Deployment Recommendation:**
Users should do a "warm-up run" when their server starts to ensure the Python environment is already initialized before handling requests. This prevents the first request from timing out during the initial download/install phase.

**Example warm-up pattern:**
```elixir
# In your application startup or release scripts
def warm_up_margarine do
  # This triggers Pythonx initialization if not already done
  Margarine.check_environment()
  # Or run a quick test generation with minimal steps
end
```

**TODO: Revisit this recommendation when writing the README to ensure it's still accurate and covers any edge cases discovered during development.**

## Scheduler & Pipeline Architecture

**Goal (from `imagine`):** Create generalized, composable schedulers and pipelines that allow mixing and matching of scheduling algorithms with different models.

**Design Principles:**
- Schedulers are pure Nx implementations (no Python dependency)
- Schedulers implement a common behavior/protocol
- Pipelines coordinate between schedulers and model inference
- Easy to add new schedulers without touching model code
- Users can swap schedulers via configuration

**Example Schedulers:**
- `FluxEuler` - Rectified flow Euler scheduler (current)
- `DDIM` - Denoising Diffusion Implicit Models (future)
- `DPMSolverMultistep` - DPM++ scheduler (future)
- Custom user-defined schedulers

**Reference Implementation:**
See `~/code/imagine` for working examples of scheduler/pipeline separation and composability. Key files:
- `lib/imagine/scheduler.ex` - Behavior definition
- `lib/imagine/schedulers/flux_euler.ex` - Pure Nx implementation
- `lib/imagine/pipeline.ex` - Coordination logic

This architecture enables:
- Testing schedulers independently of model loading
- Swapping schedulers without changing generation code
- Community contributions of new scheduling algorithms
- Research and experimentation with novel sampling methods

## Target User Experience

```elixir
# Simple case
{:ok, image} = Margarine.generate("a red panda eating bamboo")

# Advanced case with options
{:ok, image} = Margarine.generate("a red panda eating bamboo",
  model: :flux_schnell,
  steps: 4,
  guidance_scale: 3.5,
  seed: 42,
  size: {1024, 1024},
  backend: :emlx
)

# Streaming intermediate results
Margarine.generate_stream("a red panda eating bamboo", steps: 20)
|> Stream.each(fn {:step, n, image} ->
  IO.puts("Step #{n}/20")
  Margarine.save(image, "step_#{n}.png")
end)
|> Stream.run()

# Batch generation
prompts = ["red panda", "blue panda", "green panda"]
{:ok, images} = Margarine.generate_batch(prompts, model: :flux_schnell)

# Image-to-image
{:ok, image} = Margarine.generate("turn this into a painting",
  source_image: "photo.png",
  strength: 0.7
)
```

## Project Structure

```
margarine/
├── lib/
│   ├── margarine.ex                 # Main API module
│   ├── margarine/
│   │   ├── application.ex          # OTP application (Python process supervision)
│   │   ├── config.ex               # Configuration handling
│   │   ├── models/
│   │   │   ├── flux.ex             # FLUX model integration
│   │   │   ├── stable_diffusion.ex # SD integration (future)
│   │   │   └── model.ex            # Model behavior
│   │   ├── python/
│   │   │   ├── server.ex           # Pythonx integration layer
│   │   │   └── runtime.ex          # Python process management
│   │   ├── pipeline.ex             # Generation pipeline coordination
│   │   ├── image.ex                # Image encoding/decoding
│   │   └── telemetry.ex            # Instrumentation
│   └── margarine_web/              # Optional LiveView demo (separate)
├── priv/
│   └── python/
│       ├── flux_server.py          # FLUX inference server
│       ├── requirements.txt        # Python dependencies
│       └── utils.py                # Helper functions
├── test/
├── examples/                       # Example scripts
├── CLAUDE.md                       # This file
└── mix.exs
```

## Technical Architecture

### Phase 1: FLUX Integration (MVP)

**Components:**

1. **Python Server** (`priv/python/flux_server.py`)
   - Loads FLUX model on startup
   - Exposes inference API via Pythonx
   - Handles tensor operations with PyTorch/MPS or CUDA
   - Returns numpy arrays for image data

2. **Elixir Wrapper** (`lib/margarine/python/server.ex`)
   - Manages Python process lifecycle
   - Converts Elixir data → Python via Pythonx
   - Handles zero-copy tensor transfer
   - Error handling and retries

3. **Pipeline Coordinator** (`lib/margarine/pipeline.ex`)
   - Orchestrates generation workflow
   - Handles streaming intermediate results
   - Manages batching and concurrency

4. **Image Processing** (`lib/margarine/image.ex`)
   - Converts between Nx tensors and image formats
   - Uses Vix/Image for PNG/JPEG encoding
   - Handles image preprocessing for img2img

5. **Public API** (`lib/margarine.ex`)
   - Clean, Elixir-idiomatic interface
   - Documentation and examples
   - Type specs for all public functions

### Model Download Strategy

**Options to consider:**

1. **Hugging Face Cache** (recommended for MVP)
   - Use `transformers` library's built-in caching
   - Models download to `~/.cache/huggingface/`
   - Automatic on first use, user just needs HF token
   - Example: `from_pretrained("black-forest-labs/FLUX.1-schnell")`

2. **Priv Directory** (alternative)
   - Download models to `priv/models/`
   - More control, but larger app size
   - Good for air-gapped deployments

3. **User-Specified Path** (power users)
   - Allow config like `model_cache_dir: "/data/models"`
   - Useful for containers, shared storage

**MVP Decision:** Use Hugging Face cache, add config option later.

### Backend Support

Following Arcana's pattern, users configure their Nx backend in their own application config. Margarine works with whatever backend is configured.

**EMLX (Apple Silicon/Metal):**
```elixir
# config/config.exs
config :nx,
  default_backend: EMLX.Backend,
  default_defn_options: [compiler: EMLX]

# mix.exs
{:emlx, "~> 0.1"}
```

**EXLA (CUDA/ROCm/CPU via XLA):**
```elixir
# config/config.exs
config :nx,
  default_backend: EXLA.Backend,
  default_defn_options: [compiler: EXLA]

# mix.exs
{:exla, "~> 0.9"}
```

**Torchx (PyTorch backend):**
```elixir
# config/config.exs
config :nx,
  default_backend: {Torchx.Backend, device: :cpu}  # or :cuda, :mps

# mix.exs
{:torchx, "~> 0.7"}
```

**Why this approach?**
- Users have full control over their compute backend
- Margarine doesn't force specific hardware requirements
- Backend dependencies are user-managed (lighter dependency tree)
- Works with any Nx-compatible backend (future-proof)

### Error Handling

**Common scenarios:**
- Model not found / download failed
- CUDA/MPS out of memory
- Invalid prompt or parameters
- Python process crash
- Image encoding/decoding errors

**Strategy:**
- Return `{:ok, result}` or `{:error, reason}` tuples
- Detailed error messages with recovery suggestions
- Telemetry events for monitoring
- Graceful degradation where possible

## Development Phases

### Phase 1: FLUX Schnell (Quick Wins - 2-3 weeks)

**Goal:** Get basic text-to-image working with FLUX Schnell (4-step model)

**Beads:**
1. `margarine-xxx`: Project setup - dependencies, config, basic structure
2. `margarine-xxx`: Python FLUX server - load model, basic inference
3. `margarine-xxx`: Pythonx integration - process management, data transfer
4. `margarine-xxx`: Image encoding/decoding - Nx tensors ↔ PNG
5. `margarine-xxx`: Public API - `Margarine.generate/2`
6. `margarine-xxx`: Testing - unit tests, integration tests
7. `margarine-xxx`: Documentation - README, API docs, examples
8. `margarine-xxx`: Hex package preparation - publish to Hex.pm

**Deliverables:**
- Working library on Hex.pm
- Example scripts
- Blog post / demo video
- GitHub repo with stars ⭐

### Phase 2: Advanced FLUX Features (1-2 weeks)

**Beads:**
1. `margarine-xxx`: FLUX Dev model support (higher quality, more steps)
2. `margarine-xxx`: Streaming intermediate results
3. `margarine-xxx`: Batch generation
4. `margarine-xxx`: Seed control for reproducibility
5. `margarine-xxx`: Custom schedulers (DDIM, Euler, etc.)
6. `margarine-xxx`: Telemetry and instrumentation
7. `margarine-xxx`: Performance optimizations

### Phase 3: Image-to-Image (1 week)

**Beads:**
1. `margarine-xxx`: Image loading and preprocessing
2. `margarine-xxx`: Strength parameter (denoising level)
3. `margarine-xxx`: Image-to-image pipeline
4. `margarine-xxx`: Examples and documentation

### Phase 4: Stable Diffusion (Backfill - 2-3 weeks)

**Beads:**
1. `margarine-xxx`: SD model loading and inference
2. `margarine-xxx`: Model behavior abstraction
3. `margarine-xxx`: Unified API across models
4. `margarine-xxx`: Model selection and switching
5. `margarine-xxx`: SD-specific features (negative prompts, etc.)

### Phase 5: Production Features (Ongoing)

**Nice-to-haves:**
- ControlNet integration
- LoRA support
- Inpainting/outpainting
- Upscaling
- Multi-model ensembles
- GPU memory management
- Rate limiting and queuing
- CloudFlare R2 / S3 integration for storage

## Dependencies

**Elixir (core dependencies):**
```elixir
{:nx, "~> 0.9"},                            # Required
{:pythonx, "~> 0.2"},                       # Required - Python integration via UV
{:vix, "~> 0.31"},                          # Required - Image processing
{:telemetry, "~> 1.0"},                     # Required
{:jason, "~> 1.4"},                         # Required
```

**Backend Dependencies (user chooses ONE):**
```elixir
# Apple Silicon users
{:emlx, "~> 0.1"}

# NVIDIA/AMD GPU or CPU users
{:exla, "~> 0.9"}

# PyTorch users (experimental)
{:torchx, "~> 0.7"}
```

**Note:** Margarine lists backends as **optional dependencies** in mix.exs. Users must add one to their own deps.

**Python (requirements.txt):**
```
torch>=2.0.0
transformers>=4.30.0
diffusers>=0.21.0
pillow>=10.0.0
numpy>=1.24.0
```

## Configuration

**Application config (config/config.exs):**
```elixir
config :margarine,
  # Backend selection
  backend: {EMLX.Backend, device: :gpu},

  # Model cache directory
  model_cache_dir: nil,  # nil = use HF cache

  # Python environment
  python_executable: "python3",

  # Generation defaults
  default_model: :flux_schnell,
  default_steps: 4,
  default_guidance_scale: 3.5,
  default_size: {1024, 1024},

  # Performance
  max_concurrent: 1,  # Most GPUs can only run one model at a time
  timeout: 60_000,    # 60 seconds per generation

  # Telemetry
  enable_telemetry: true
```

## Testing Strategy

1. **Unit tests**: Individual modules, mocked Python calls
2. **Integration tests**: Real Python server, real models (CI: CPU only)
3. **Manual tests**: GPU backends, visual quality checks
4. **Example scripts**: Double as smoke tests

## Documentation Plan

1. **README.md**: Quick start, installation, basic examples
2. **HexDocs**: Full API documentation with examples
3. **guides/**: Step-by-step tutorials
   - Getting started
   - Advanced usage
   - Backend configuration
   - Production deployment
4. **Blog post**: "Introducing Margarine: AI Image Generation for Elixir"
5. **Demo video**: 2-3 minute screencast

## Success Metrics

**Technical:**
- ✅ Generate 1024x1024 image in <5s on M4/MPS
- ✅ Generate 1024x1024 image in <2s on RTX 4090/CUDA
- ✅ Zero-copy tensor transfer working
- ✅ Streaming intermediate results
- ✅ Proper error handling and recovery

**Community:**
- 🎯 100+ GitHub stars in first month
- 🎯 10+ downloads per day on Hex.pm
- 🎯 Used in at least 3 real projects
- 🎯 Mentioned on Elixir Forum, Reddit, Twitter
- 🎯 Lightning talk at ElixirConf or meetup

## Open Questions for Discussion

1. **Name**: Are we happy with "Margarine"? Alternatives?
2. **Model scope**: Start with just FLUX Schnell, or include FLUX Dev in Phase 1?
3. **Image format**: Default to PNG, or support JPEG/WebP too?
4. **Streaming**: HTTP streaming for LiveView integration? Or just in-memory?
5. **Licensing**: MIT? Apache 2.0? (Consider model licenses too)
6. **HuggingFace token**: Require env var, or optional config?
7. **Python version**: Target 3.11+? Support 3.9+?

## Notes from GenericJam Image Generation Experiments

**What worked well:**
- Pythonx for zero-copy integration
- EMLX backend on Apple Silicon
- Nx tensor manipulation
- Streaming results to LiveView

**Pain points:**
- Model download is unclear to users
- Backend configuration is confusing
- Error messages from Python are cryptic
- Memory management needs attention

**Improvements for Margarine:**
- Better error messages
- Clear documentation on model setup
- Automatic backend detection
- Memory usage monitoring

## Next Steps

1. Review this plan together
2. Create initial beads for Phase 1
3. Set up project structure
4. Start with Python server + basic inference
5. Iterate!

---

**Remember:** Ship early, ship often. Phase 1 is the MVP - get it out there and iterate based on feedback!
