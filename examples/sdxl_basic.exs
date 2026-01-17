#!/usr/bin/env elixir

# SDXL Basic Example
#
# This script demonstrates basic SDXL image generation with Margarine.
# SDXL provides photorealistic quality and is great for detailed images.
#
# Prerequisites:
# - Mix dependencies installed: `mix deps.get`
# - Nx backend configured (EMLX for Apple Silicon, EXLA for NVIDIA/AMD)
#
# Run with: `elixir examples/sdxl_basic.exs`
#
# Note: First run will download SDXL models (~7GB).
# This takes 2-5 minutes. Subsequent runs are instant!

Mix.install([
  {:margarine, path: "."},
  {:emlx, "~> 0.1"}  # Change to {:exla, "~> 0.9"} for NVIDIA/AMD
])

# Configure Nx backend (if not in config/config.exs)
Application.put_env(:nx, :default_backend, EMLX.Backend)
Application.put_env(:nx, :default_defn_options, compiler: EMLX)

IO.puts("\n🎨 Margarine - SDXL Basic Example\n")
IO.puts("Generating photorealistic image with SDXL...\n")

# SDXL Turbo: Fast, 1-step generation
prompt = "a serene mountain lake at sunset, photorealistic, highly detailed"
IO.puts("Prompt: #{prompt}")
IO.puts("Model: SDXL Turbo (1 step, fast)\n")

case Margarine.generate(prompt, model: :sdxl_turbo, steps: 1, seed: 42) do
  {:ok, image} ->
    # Print image info
    {height, width, channels} = Nx.shape(image)
    type = Nx.type(image)

    IO.puts("✓ Image generated successfully!")
    IO.puts("  Shape: #{height}x#{width}x#{channels}")
    IO.puts("  Type: #{inspect(type)}")

    # Save to file
    output_path = "output_sdxl_basic.png"
    case Margarine.Image.save(image, output_path) do
      :ok ->
        IO.puts("  Saved to: #{output_path}")

      {:error, reason} ->
        IO.puts("  Failed to save: #{reason}")
    end

  {:error, reason} ->
    IO.puts("\n✗ Generation failed: #{reason}")
    IO.puts("\nIf this is your first run, please wait 2-5 minutes for setup.")
    IO.puts("If the error persists, check:")
    IO.puts("  - Available RAM (need 16GB+)")
    IO.puts("  - Internet connection (for model download)")
    IO.puts("  - Nx backend configuration")

    System.halt(1)
end

IO.puts("\n" <> String.duplicate("=", 60))
IO.puts("Want higher quality? Try SDXL Base:")
IO.puts(String.duplicate("=", 60))
IO.puts(~S"""

# SDXL Base (higher quality, slower)
{:ok, image} = Margarine.generate(
  "#{prompt}",
  model: :sdxl_base,
  steps: 20,           # More steps = better quality
  guidance_scale: 7.5, # Higher = stronger prompt adherence
  seed: 42
)

Margarine.Image.save(image, "output_sdxl_base.png")
""")

IO.puts("\n✨ Done!\n")
