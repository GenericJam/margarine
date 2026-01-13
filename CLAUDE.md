# Margarine - Elixir Image Generation Library

**"I Can't Believe It's Not Butter... I Mean Python!"**

---

## Development Tools

### Semantic Code Search with C-K (seek)

We have `ck` (C-K, pronounced "seek") available for semantic grep - a way to cast a wider net when searching through code. When searching for patterns, consider using `ck` for semantic understanding beyond simple text matching.

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
