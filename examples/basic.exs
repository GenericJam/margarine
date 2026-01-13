#!/usr/bin/env elixir

# Basic Margarine Example
#
# This script demonstrates the simplest way to generate images with Margarine.
#
# Prerequisites:
# - Mix dependencies installed: `mix deps.get`
# - Nx backend configured (EMLX for Apple Silicon, EXLA for NVIDIA/AMD)
#
# Run with: `elixir examples/basic.exs`
#
# Note: First run will download Python (~100MB) and models (~12GB).
# This takes 2-5 minutes. Subsequent runs are instant!

Mix.install([
  {:margarine, path: "."},
  {:emlx, "~> 0.1"}  # Change to {:exla, "~> 0.9"} for NVIDIA/AMD
])

# Configure Nx backend (if not in config/config.exs)
Application.put_env(:nx, :default_backend, EMLX.Backend)
Application.put_env(:nx, :default_defn_options, compiler: EMLX)

IO.puts("\n🎨 Margarine - Basic Example\n")
IO.puts("Generating image from text prompt...\n")

# Simple generation with defaults
prompt = "a red panda eating bamboo in a forest"
IO.puts("Prompt: #{prompt}")

case Margarine.generate(prompt) do
  {:ok, image} ->
    # Print image info
    {height, width, channels} = Nx.shape(image)
    type = Nx.type(image)

    IO.puts("\n✓ Image generated successfully!")
    IO.puts("  Shape: #{height}x#{width}x#{channels}")
    IO.puts("  Type: #{inspect(type)}")

    # Save to file
    output_path = "output_basic.png"
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

IO.puts("\n✨ Done!\n")
