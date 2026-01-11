# Margarine

<p align="center">
  <img src="logo.png" alt="Margarine Logo" width="400"/>
</p>

<p align="center">
  <strong>"I Can't Believe It's Not Butter... I Mean Python!"</strong>
</p>

---

AI-powered image generation for Elixir using FLUX and Stable Diffusion.

## Features

- 🎨 **FLUX Integration** - Fast, high-quality image generation with FLUX Schnell
- ⚡ **Zero-Copy Performance** - Pythonx integration for efficient tensor transfer
- 🍎 **Apple Silicon Support** - Optimized for M-series Macs with EMLX/Metal
- 🔥 **CUDA Support** - NVIDIA GPU acceleration via EXLA
- 📊 **Streaming Results** - Watch your images form step-by-step
- 🧪 **Production Ready** - Comprehensive tests, telemetry, and error handling

## Quick Start

```elixir
# Simple text-to-image
{:ok, image} = Margarine.generate("a red panda eating bamboo")

# Advanced options
{:ok, image} = Margarine.generate("a red panda eating bamboo",
  model: :flux_schnell,
  steps: 4,
  guidance_scale: 3.5,
  seed: 42,
  size: {1024, 1024}
)
```

## Installation

Add `margarine` to your list of dependencies in `mix.exs`:

```elixir
def deps do
  [
    {:margarine, "~> 0.1.0"}
  ]
end
```

## Requirements

- Elixir 1.14+
- Python 3.11+
- PyTorch 2.0+
- CUDA-capable GPU or Apple Silicon Mac (M1/M2/M3/M4)

## Documentation

Full documentation available at [hexdocs.pm/margarine](https://hexdocs.pm/margarine).

See [CLAUDE.md](CLAUDE.md) for development notes and architecture details.

## Development

This project follows strict **Test-Driven Development (TDD)** practices:
- Tests written before implementation
- 80%+ code coverage required
- Run tests: `mix test`
- Run with coverage: `mix test --cover`
- Integration tests: `mix test --only integration`

## License

MIT

## Credits

Logo generated with FLUX - a perfect example of what this library can do!
