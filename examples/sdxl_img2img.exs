#!/usr/bin/env elixir

# SDXL Image-to-Image Example
#
# This demonstrates SDXL img2img functionality by transforming existing images.
# Shows different denoising strengths and their effects.
#
# Run with: `elixir examples/sdxl_img2img.exs`

Mix.install([
  {:margarine, path: "."},
  {:emlx, "~> 0.1"}
])

# Configure Nx backend
Application.put_env(:nx, :default_backend, EMLX.Backend)
Application.put_env(:nx, :default_defn_options, compiler: EMLX)

defmodule SdxlImg2ImgExample do
  @moduledoc false

  def run do
    IO.puts("\n🎨 Margarine - SDXL IMG2IMG Example\n")

    # First, generate a base image to transform
    base_image = "output_sdxl_img2img_base.png"

    unless File.exists?(base_image) do
      IO.puts("Step 1: Generating base image...")
      generate_base_image(base_image)
    else
      IO.puts("Using existing base image: #{base_image}")
    end

    IO.puts("\nStep 2: Applying transformations with different strengths...\n")

    # Test 1: Subtle style change (30% modification)
    IO.puts(String.duplicate("=", 80))
    IO.puts("TEST 1: Subtle style change (strength: 0.3)")
    IO.puts("Effect: Keeps most of the original, adds artistic flair")
    IO.puts(String.duplicate("=", 80))

    transform_image(
      "watercolor painting style, soft brush strokes",
      base_image,
      0.3,
      "output_sdxl_img2img_watercolor.png"
    )

    # Test 2: Moderate transformation (60% modification)
    IO.puts("\n" <> String.duplicate("=", 80))
    IO.puts("TEST 2: Moderate transformation (strength: 0.6)")
    IO.puts("Effect: Significant changes while preserving composition")
    IO.puts(String.duplicate("=", 80))

    transform_image(
      "oil painting, impressionist style, vibrant colors",
      base_image,
      0.6,
      "output_sdxl_img2img_oil.png"
    )

    # Test 3: Heavy modification (80% change)
    IO.puts("\n" <> String.duplicate("=", 80))
    IO.puts("TEST 3: Heavy modification (strength: 0.8)")
    IO.puts("Effect: Major transformation, only rough composition remains")
    IO.puts(String.duplicate("=", 80))

    transform_image(
      "cyberpunk city at night, neon lights, futuristic",
      base_image,
      0.8,
      "output_sdxl_img2img_cyberpunk.png"
    )

    # Test 4: Complete regeneration (strength=1.0)
    IO.puts("\n" <> String.duplicate("=", 80))
    IO.puts("TEST 4: Complete regeneration (strength: 1.0)")
    IO.puts("Effect: Equivalent to text2img, ignores original image")
    IO.puts(String.duplicate("=", 80))

    transform_image(
      "a majestic dragon flying over mountains",
      base_image,
      1.0,
      "output_sdxl_img2img_dragon.png"
    )

    IO.puts("\n✨ All transformations complete!\n")
    IO.puts("Generated images:")
    IO.puts("  - #{base_image} (original)")
    IO.puts("  - output_sdxl_img2img_watercolor.png (30% change - subtle)")
    IO.puts("  - output_sdxl_img2img_oil.png (60% change - moderate)")
    IO.puts("  - output_sdxl_img2img_cyberpunk.png (80% change - heavy)")
    IO.puts("  - output_sdxl_img2img_dragon.png (100% change - new image)")
    IO.puts("\nCompare them to see how denoising strength affects the output!\n")
  end

  defp generate_base_image(output_path) do
    prompt = "a peaceful garden with flowers and a fountain, photorealistic"
    IO.puts("  Prompt: #{prompt}")

    start_time = System.monotonic_time(:millisecond)

    case Margarine.generate(prompt, model: :sdxl_turbo, steps: 1, seed: 42, size: {512, 512}) do
      {:ok, image} ->
        elapsed = System.monotonic_time(:millisecond) - start_time
        elapsed_sec = Float.round(elapsed / 1000, 1)

        case Margarine.Image.save(image, output_path) do
          :ok ->
            IO.puts("  ✓ Generated in #{elapsed_sec}s")
            IO.puts("  ✓ Saved to: #{output_path}\n")

          {:error, reason} ->
            IO.puts("  ✗ Save failed: #{reason}")
            System.halt(1)
        end

      {:error, reason} ->
        IO.puts("  ✗ Generation failed: #{reason}")
        System.halt(1)
    end
  end

  defp transform_image(prompt, init_image, strength, output_file) do
    IO.puts("\nPrompt: #{prompt}")
    IO.puts("Denoising strength: #{strength}")
    IO.puts("Output: #{output_file}\n")

    start_time = System.monotonic_time(:millisecond)

    result =
      Margarine.img2img(
        prompt,
        init_image,
        denoising_strength: strength,
        model: :sdxl_turbo,
        steps: 1,
        seed: 42
      )

    elapsed = System.monotonic_time(:millisecond) - start_time
    elapsed_sec = Float.round(elapsed / 1000, 1)

    case result do
      {:ok, image} ->
        {h, w, c} = Nx.shape(image)

        case Margarine.Image.save(image, output_file) do
          :ok ->
            IO.puts("✓ Generated in #{elapsed_sec}s")
            IO.puts("✓ Image: #{h}x#{w}x#{c}")
            IO.puts("✓ Saved to: #{output_file}")

          {:error, reason} ->
            IO.puts("✗ Save failed: #{reason}")
        end

      {:error, reason} ->
        IO.puts("✗ Generation failed: #{reason}")
    end
  end
end

# Run the example
SdxlImg2ImgExample.run()
