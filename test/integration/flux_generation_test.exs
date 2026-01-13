defmodule Margarine.Integration.FluxGenerationTest do
  use ExUnit.Case

  @moduledoc """
  Integration tests for real FLUX image generation.

  These tests verify the complete end-to-end pipeline:
  1. Python environment initialization
  2. Model loading (or using pre-loaded model)
  3. Text prompt encoding
  4. Denoising loop with scheduler
  5. Image decoding to tensor
  6. Validation of output format

  **IMPORTANT:** These tests are marked with `@tag :integration` and
  excluded by default because they:
  - Require downloading ~12GB FLUX model (first run)
  - Need significant RAM/VRAM (16GB+ RAM or 12GB+ VRAM)
  - Take 5-30 seconds per image (depending on hardware)

  ## Running Integration Tests

      # Run only integration tests
      mix test --only integration

      # Run all tests including integration
      mix test --include integration

  ## Prerequisites

  - Sufficient memory (16GB+ RAM or 12GB+ VRAM)
  - HuggingFace access token (optional, for faster downloads)
  - Stable internet connection (first run)

  ## Test Strategy

  We use actual FLUX generation but with minimal parameters to keep
  tests fast:
  - flux_schnell (4 steps, faster)
  - Small images (512x512)
  - Simple prompts

  This verifies the complete pipeline works without taking forever.
  """

  describe "FLUX Schnell generation" do
    @describetag :integration
    @describetag timeout: 120_000  # 2 minutes per test (generous for first run + model load)

    test "generates valid image from text prompt" do
      # Simple test prompt
      prompt = "a red circle on white background"

      # Minimal settings for speed
      opts = [
        model: :flux_schnell,
        steps: 4,
        size: {512, 512},
        seed: 42  # Reproducibility
      ]

      # Generate image
      result = Margarine.generate(prompt, opts)

      # Verify successful generation
      assert {:ok, image} = result
      assert %Nx.Tensor{} = image

      # Verify tensor shape and type
      assert Nx.shape(image) == {512, 512, 3}
      assert Nx.type(image) == {:u, 8}

      # Verify image is not blank (has some variation)
      # A blank image would have all same values
      min_val = Nx.reduce_min(image) |> Nx.to_number()
      max_val = Nx.reduce_max(image) |> Nx.to_number()
      assert max_val > min_val, "Image appears blank (no variation in pixel values)"

      # Verify pixel values are in valid range
      assert min_val >= 0
      assert max_val <= 255
    end

    test "respects seed for reproducibility" do
      prompt = "a blue square"
      opts = [model: :flux_schnell, steps: 4, size: {512, 512}, seed: 123]

      # Generate twice with same seed
      {:ok, image1} = Margarine.generate(prompt, opts)
      {:ok, image2} = Margarine.generate(prompt, opts)

      # Should be identical
      assert Nx.equal(image1, image2) |> Nx.all() |> Nx.to_number() == 1
    end

    test "produces different results with different seeds" do
      prompt = "a green triangle"
      opts1 = [model: :flux_schnell, steps: 4, size: {512, 512}, seed: 1]
      opts2 = [model: :flux_schnell, steps: 4, size: {512, 512}, seed: 2]

      {:ok, image1} = Margarine.generate(prompt, opts1)
      {:ok, image2} = Margarine.generate(prompt, opts2)

      # Should be different
      refute Nx.equal(image1, image2) |> Nx.all() |> Nx.to_number() == 1
    end
  end

  describe "error handling" do
    @describetag :integration
    @describetag timeout: 120_000

    test "handles memory constraints gracefully" do
      # Try to generate a huge image that might exceed memory
      # This should fail gracefully, not crash the VM
      prompt = "test"
      opts = [model: :flux_schnell, size: {4096, 4096}]

      result = Margarine.generate(prompt, opts)

      # Should either succeed or fail gracefully
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end
  end
end
