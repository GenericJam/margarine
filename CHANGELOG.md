# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-01-12

### Added

#### Core Functionality
- **FLUX Integration** - Support for FLUX Schnell and FLUX Dev models
- **Public API** - `Margarine.generate/1` and `Margarine.generate/2` for text-to-image generation
- **Zero-Copy Tensor Transfer** - Efficient Pythonx integration for Nx ↔ PyTorch tensors
- **Automatic Python Installation** - UV-based dependency management, zero user configuration
- **Memory Safety** - Pre-flight RAM checks to prevent OOM crashes
- **Reproducible Generation** - Seed control for deterministic results

#### Configuration & Options
- Model selection (`:flux_schnell`, `:flux_dev`)
- Customizable steps (4 for Schnell, 28 for Dev)
- Guidance scale control
- Custom image dimensions (must be divisible by 8)
- Random seed support for reproducibility

#### Architecture
- **Pipeline Coordinator** - Orchestrates generation workflow with validation
- **Pure Nx Schedulers** - FluxEuler scheduler implemented in pure Elixir/Nx
- **Modular Design** - Swappable schedulers and composable pipelines
- **Image Module** - Nx tensor ↔ PNG conversion using Vix
- **Config Module** - Model-specific defaults and validation

#### Testing
- 119 unit tests with 75.3% code coverage
- 4 integration tests for end-to-end validation (excluded by default)
- Test-Driven Development (TDD) throughout
- Comprehensive parameter validation tests
- Memory safety tests

#### Documentation
- Complete README with installation, usage, and troubleshooting
- Example scripts (`examples/basic.exs`, `examples/advanced.exs`)
- HexDocs generation for all public APIs
- Architecture documentation in CLAUDE.md
- Inline code documentation with `@moduledoc` and `@doc`

#### Developer Experience
- Credo linting with strict rules
- Dialyzer type specs on all public functions
- ExUnit async tests for fast feedback
- Coverage reporting with `mix test --cover`

### Technical Details

#### Dependencies
- Elixir 1.14+ required
- Nx ~> 0.9 for numerical computing
- Pythonx ~> 0.2 for Python integration
- Vix ~> 0.31 for image processing
- Optional backends: EMLX (Apple Silicon) or EXLA (CUDA/CPU)

#### Python Requirements (auto-installed)
- Python 3.11+
- PyTorch 2.0+
- Diffusers 0.21+
- Transformers 4.30+
- Accelerate, SafeTensors, Protobuf, SentencePiece

#### Memory Requirements
- FLUX Schnell: ~12GB VRAM (GPU) or ~16GB RAM (CPU)
- FLUX Dev: ~12GB VRAM (GPU) or ~16GB RAM (CPU)
- First run downloads ~12GB FLUX model

### Known Limitations

- Python server integration not yet complete (returns error for now)
- No streaming intermediate results (planned for Phase 2)
- No batch generation (planned for Phase 2)
- No image-to-image support (planned for Phase 3)
- No Stable Diffusion support (planned for Phase 4)

### Development Stats

- **Lines of Code**: ~2,500 (excluding tests)
- **Test Coverage**: 75.3%
- **Tests**: 119 unit tests, 4 integration tests
- **Commits**: 10+ across 10 beads (TDD workflow)
- **Development Time**: ~4 hours (following TDD practices)

### Contributors

Built with Test-Driven Development practices by the Margarine team.

### Acknowledgments

- **Black Forest Labs** - For the amazing FLUX models
- **Pythonx** - For enabling seamless Python integration in Elixir
- **Nx Team** - For the Elixir numerical computing foundation
- **Bumblebee** - For inspiration and ML patterns in Elixir

---

## [0.2.0] - 2026-01-16

### Added

#### SDXL Integration
- **SDXL Base Model** - High-quality photorealistic generation (20 steps, guidance scale 7.5)
- **SDXL Turbo Model** - Ultra-fast photorealistic generation (1 step)
- **Dual CLIP Encoders** - CLIP-ViT-L + CLIP-ViT-G with projection for SDXL text encoding
- **DDIM Scheduler** - Pure Nx implementation of Denoising Diffusion Implicit Models
- **Time ID Conditioning** - SDXL-specific conditioning for resolution and crop parameters
- **Model Selection** - Choose between FLUX (artistic) and SDXL (photorealistic) for different use cases

#### Image-to-Image Transformation
- **`Margarine.img2img/3` API** - Transform existing images with text prompts
- **Denoising Strength Parameter** - Control transformation amount (0.0 = no change, 1.0 = complete regeneration)
- **Works with All Models** - Both FLUX and SDXL support img2img transformation
- **VAE Encoding** - Convert images to latent space for transformation
- **Noise Addition** - Scheduler-based noise mixing for controlled modifications

#### Image Processing Enhancements
- **RGBA Support** - Automatic conversion of RGBA images to RGB (alpha channel removal)
- **Non-Square Images** - Smart resize preserves content with minimal cropping/padding
- **Automatic Dimension Rounding** - Auto-round to nearest multiple of 8 for VAE compatibility
- **Dimension Auto-Detection** - img2img defaults to original image size (rounded to multiple of 8)
- **Intelligent Resize Strategy** - Scale-then-adjust approach preserves image content

#### Documentation & Examples
- **SDXL Examples** - New example scripts: `examples/sdxl_basic.exs`, `examples/sdxl_img2img.exs`
- **Updated README** - Comprehensive SDXL and img2img documentation
- **Model Comparison Guide** - FLUX vs SDXL usage recommendations
- **IMG2IMG Strength Guide** - Detailed explanation of denoising strength effects
- **Updated Livebook** - SDXL section in getting started notebook

#### Testing
- **13 SDXL Integration Tests** - Comprehensive test coverage for SDXL text2img and img2img
- **Edge Case Tests** - RGBA handling, non-square images, dimension rounding
- **Total: 148 Tests** - Up from 119 tests in v0.1.0
- **All Tests Passing** - 100% pass rate on integration test suite

### Fixed
- **Automatic UV Installation** - Added Pythonx configuration to auto-install UV package manager
  - No more "UV not found" errors on fresh installations
  - UV downloads and installs automatically at compile time
  - Bundled with Elixir releases for offline deployment
  - Added `config/config.exs` with `uv_version: "0.8.5"`

### Changed
- **Updated Memory Requirements** - Added SDXL models (~7GB vs FLUX's ~12GB)
- **Configuration Options** - Extended to include SDXL models and img2img parameters
- **Package Description** - Now includes "text-to-image and image-to-image"

### Technical Details

#### New Modules
- `Margarine.Pipelines.Sdxl` - SDXL generation pipeline (unified text2img and img2img)
- `Margarine.Python.SdxlPythonxServer` - SDXL model server GenServer
- `Margarine.Schedulers.DDIM` - Pure Nx DDIM scheduler implementation
- `priv/python/sdxl_pythonx.py` - Python SDXL inference module

#### Memory Requirements
- **SDXL Turbo**: ~7GB VRAM (GPU) or ~12GB RAM (CPU)
- **SDXL Base**: ~7GB VRAM (GPU) or ~12GB RAM (CPU)
- First run downloads ~7GB SDXL models (in addition to FLUX if already installed)

#### Key Insights
- **Everything is IMG2IMG** - Text2img is just img2img starting from pure noise
- **Unified Pipeline** - Same denoising loop for both text2img and img2img
- **Shared Latent Space** - FLUX and SDXL use same VAE, enabling future cross-model mixing

## [Unreleased]

### Planned for Phase 3: Inpainting
- SDXL Inpaint model support
- Mask preprocessing utilities
- Inpainting pipeline
- Interactive mask editing

### Planned for Phase 4: Advanced Features
- Streaming intermediate results during generation
- Batch generation for multiple prompts
- LoRA support
- ControlNet integration
- Additional schedulers (DPM++, etc.)
- Performance optimizations
- Cross-model mixing (FLUX + SDXL in same generation)

[0.2.0]: https://github.com/GenericJam/margarine/releases/tag/v0.2.0
[0.1.0]: https://github.com/GenericJam/margarine/releases/tag/v0.1.0
