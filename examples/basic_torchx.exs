#!/usr/bin/env elixir

# Basic Margarine Example with Torchx Backend
#
# This script demonstrates using Margarine with Torchx (CPU-only)
#
# Run with: `elixir examples/basic_torchx.exs`

Mix.install(
  [
    {:margarine, path: "."},
    {:torchx, "~> 0.10"}
  ],
  config: [
    nx: [default_backend: {Torchx.Backend, device: :mps}]
  ]
)

# IMPORTANT: Force CPU device for Torchx compatibility
# Torchx downloads CPU-only libtorch by default
# Without this, Margarine will try to use MPS on macOS and crash
System.put_env("MARGARINE_DEVICE", "mps")

IO.puts("\n🎨 Margarine - Basic Example (Torchx Backend)\n")
IO.puts("Note: Using MPS device (Apple Silicon GPU)\n")
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
    output_path = "output_basic_torchx.png"
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
IO.puts("\nNote: Torchx with MPS shows comparable performance to EMLX.")
IO.puts("Torchx is experimental but promising for cross-platform development.")
