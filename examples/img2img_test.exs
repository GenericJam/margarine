#!/usr/bin/env elixir

# IMG2IMG Test Script
#
# This demonstrates img2img functionality by modifying an existing image.
#
# Run with: `elixir examples/img2img_test.exs`

Mix.install([
  {:margarine, path: "."},
  {:emlx, "~> 0.1"}
])

# Configure Nx backend
Application.put_env(:nx, :default_backend, EMLX.Backend)
Application.put_env(:nx, :default_defn_options, compiler: EMLX)

defmodule Img2ImgTest do
  @moduledoc false

  def run do
    IO.puts("\n🎨 Margarine - IMG2IMG Test\n")

    # Check if we have the basic output to use as init image
    init_image = "output_basic.png"

    unless File.exists?(init_image) do
      IO.puts("❌ Error: #{init_image} not found!")
      IO.puts("Please run `elixir examples/basic.exs` first to generate an image.\n")
      System.halt(1)
    end

    IO.puts("Using init image: #{init_image}")
    IO.puts("This was generated with prompt: 'a red panda eating bamboo in a forest'\n")

    # Test 1: Subtle change (30% noise)
    IO.puts("=" <> String.duplicate("=", 79))
    IO.puts("TEST 1: Subtle style change (denoising_strength: 0.3)")
    IO.puts(String.duplicate("=", 80))

    test_img2img(
      "turn into a watercolor painting",
      init_image,
      0.3,
      "output_img2img_watercolor.png"
    )

    # Test 2: Moderate change (70% noise)
    IO.puts("\n" <> String.duplicate("=", 80))
    IO.puts("TEST 2: Moderate transformation (denoising_strength: 0.7)")
    IO.puts(String.duplicate("=", 80))

    test_img2img(
      "convert to anime style with vibrant colors",
      init_image,
      0.7,
      "output_img2img_anime.png"
    )

    # Test 3: Complete regeneration (100% noise - should be like text2img)
    IO.puts("\n" <> String.duplicate("=", 80))
    IO.puts("TEST 3: Complete regeneration (denoising_strength: 1.0)")
    IO.puts("This should be equivalent to text2img!")
    IO.puts(String.duplicate("=", 80))

    test_img2img(
      "a blue dragon flying over mountains",
      init_image,
      1.0,
      "output_img2img_dragon.png"
    )

    IO.puts("\n✨ All tests complete!\n")
    IO.puts("Generated images:")
    IO.puts("  - output_img2img_watercolor.png (30% change)")
    IO.puts("  - output_img2img_anime.png (70% change)")
    IO.puts("  - output_img2img_dragon.png (100% change - like text2img)\n")
  end

  defp test_img2img(prompt, init_image, strength, output_file) do
    IO.puts("\nPrompt: #{prompt}")
    IO.puts("Denoising strength: #{strength}")
    IO.puts("Output: #{output_file}\n")

    start_time = System.monotonic_time(:millisecond)

    result =
      Margarine.img2img(
        prompt,
        init_image,
        denoising_strength: strength,
        model: :flux_schnell,
        steps: 4,
        size: {1024, 1024},
        seed: 42
      )

    elapsed_ms = System.monotonic_time(:millisecond) - start_time
    elapsed_sec = Float.round(elapsed_ms / 1000, 1)

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

# Run the test
Img2ImgTest.run()
