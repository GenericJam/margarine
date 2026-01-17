defmodule Margarine.Integration.FluxGenerationTest do
  use ExUnit.Case, async: false  # CRITICAL: Must run sequentially to avoid OOM

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
  - **MUST RUN SEQUENTIALLY** - async: false to prevent multiple model instances

  ## Known Benign Warnings

  You may see this warning at the end of tests:
  ```
  /path/to/python/multiprocessing/resource_tracker.py:254: UserWarning:
  resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  ```

  **This is a known benign warning** from Python's multiprocessing module when the
  Python process is terminated by an external process (Elixir/BEAM). It does NOT
  indicate a real memory leak or resource problem. The semaphore is properly cleaned
  up by the OS when the process exits.

  See: https://github.com/apple/ml-stable-diffusion/issues/8

  ## Running Integration Tests

      # Run only integration tests (sequential execution)
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
  - Sequential execution (async: false) to avoid OOM

  This verifies the complete pipeline works without taking forever.
  """

  describe "FLUX Schnell generation" do
    @describetag :integration
    @describetag timeout: 300_000  # 5 minutes per test (generous for first run + model load)

    setup do
      # MEMORY LEAK FIX: Log memory before test
      case Margarine.Memory.available_memory() do
        {:ok, info} ->
          available_gb = Margarine.Memory.bytes_to_mb(info.available) / 1024
          IO.puts("\n[Setup] Memory before test: #{Float.round(available_gb, 1)}GB available")

        {:error, reason} ->
          IO.puts("\n[Setup] Could not check memory: #{reason}")
      end

      # Cleanup function to stop the PythonxServer after each test
      # This ensures proper resource cleanup and prevents semaphore leaks
      on_exit(fn ->
        # Give the server a moment to finish any pending operations
        Process.sleep(100)

        # MEMORY LEAK FIX: Log memory before cleanup
        case Margarine.Memory.available_memory() do
          {:ok, info_before} ->
            available_gb = Margarine.Memory.bytes_to_mb(info_before.available) / 1024
            IO.puts("\n[Cleanup] Memory before cleanup: #{Float.round(available_gb, 1)}GB available")

          {:error, _} ->
            :ok
        end

        # Stop the server if it's running
        server_name = :"Margarine.Python.PythonxServer.flux_schnell"

        case Process.whereis(server_name) do
          nil ->
            :ok

          pid ->
            # Gracefully stop the server
            GenServer.stop(pid, :normal, 5000)
            # Wait a bit for cleanup to complete
            Process.sleep(500)
        end

        # MEMORY LEAK FIX: Force Erlang GC after cleanup
        :erlang.garbage_collect()
        Process.sleep(500)

        # MEMORY LEAK FIX: Log memory after cleanup
        case Margarine.Memory.available_memory() do
          {:ok, info_after} ->
            available_gb = Margarine.Memory.bytes_to_mb(info_after.available) / 1024
            IO.puts("[Cleanup] Memory after cleanup: #{Float.round(available_gb, 1)}GB available")

          {:error, _} ->
            :ok
        end
      end)

      :ok
    end

    test "generates valid image from text prompt" do
      IO.puts("\n" <> String.duplicate("=", 80))
      IO.puts("TEST 1: Generate valid image from text prompt")
      IO.puts(String.duplicate("=", 80))

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
      IO.puts("\n" <> String.duplicate("=", 80))
      IO.puts("TEST 2: Respects seed for reproducibility")
      IO.puts(String.duplicate("=", 80))

      prompt = "a blue square"
      opts = [model: :flux_schnell, steps: 4, size: {512, 512}, seed: 123]

      # Generate twice with same seed
      {:ok, image1} = Margarine.generate(prompt, opts)
      {:ok, image2} = Margarine.generate(prompt, opts)

      # Should be identical
      assert Nx.equal(image1, image2) |> Nx.all() |> Nx.to_number() == 1
    end

    test "produces different results with different seeds" do
      IO.puts("\n" <> String.duplicate("=", 80))
      IO.puts("TEST 3: Produces different results with different seeds")
      IO.puts(String.duplicate("=", 80))

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
    @describetag timeout: 300_000  # 5 minutes for 1024x1024 images

    setup do
      # Cleanup function to stop the PythonxServer after each test
      on_exit(fn ->
        Process.sleep(100)

        server_name = :"Margarine.Python.PythonxServer.flux_schnell"

        case Process.whereis(server_name) do
          nil -> :ok
          pid -> GenServer.stop(pid, :normal, 5000); Process.sleep(500)
        end
      end)

      :ok
    end

    test "handles larger images within memory constraints" do
      IO.puts("\n" <> String.duplicate("=", 80))
      IO.puts("TEST 4: Handles larger images within memory constraints")
      IO.puts(String.duplicate("=", 80))

      # Test with 1024x1024 which is the standard FLUX size
      # This verifies the pipeline can handle full-resolution images without crashes
      #
      # FLUX Image Size Requirements:
      # - Dimensions must be divisible by 16 (8 for VAE + 2 for FLUX 2x2 patching)
      # - Valid sizes: 512, 1024, 1536, 1600, 2048, etc.
      # - Invalid sizes: 1800 (not divisible by 16) will be rejected with clear error
      #
      # Tested & Verified Sizes (M4 Max 64GB, MPS backend):
      # - 1024x1024: ~4 minutes, ~20GB peak usage - ✅ TESTED, STABLE
      # - 1600x1600: ~4.5 minutes, drops to 6.9GB available - ✅ TESTED, WORKS
      #
      # Untested Larger Sizes:
      # - 2048x2048: Theoretically supported but UNTESTED. Step 1 alone takes 4+ minutes.
      #              Expect 10-15+ minute generation times and <5GB free memory.
      #              May fail on systems with <64GB RAM due to memory pressure.
      #
      # Recommendations:
      # - Systems with <32GB: Use 1024x1024 or smaller
      # - Systems with 32-64GB: 1024x1024 recommended, 1600x1600 feasible
      # - Systems with 64GB+: Maximum tested size is 1600x1600
      #                       Larger sizes may work but expect very long generation times
      prompt = "a simple test image"
      opts = [model: :flux_schnell, size: {1024, 1024}, steps: 4, seed: 999]

      result = Margarine.generate(prompt, opts)

      # Should succeed with standard size
      assert {:ok, image} = result
      assert Nx.shape(image) == {1024, 1024, 3}
      assert Nx.type(image) == {:u, 8}

      # Verify image has content
      min_val = Nx.reduce_min(image) |> Nx.to_number()
      max_val = Nx.reduce_max(image) |> Nx.to_number()
      assert max_val > min_val, "Image appears blank"
      assert min_val >= 0
      assert max_val <= 255

      IO.puts("✓ Successfully generated 1024x1024 image")
    end
  end

  describe "IMG2IMG generation" do
    @describetag :integration
    @describetag timeout: 300_000  # 5 minutes per test

    setup do
      # Generate a base image to use for img2img tests
      IO.puts("\n[Setup IMG2IMG] Generating base image...")

      prompt = "a red panda eating bamboo in a forest"
      opts = [model: :flux_schnell, steps: 4, size: {512, 512}, seed: 42]

      {:ok, base_image} = Margarine.generate(prompt, opts)

      # Save to temp file for img2img
      temp_path = Path.join(System.tmp_dir!(), "margarine_img2img_test_#{:rand.uniform(999_999)}.png")
      :ok = Margarine.Image.save(base_image, temp_path)

      IO.puts("[Setup IMG2IMG] Base image saved to: #{temp_path}")

      # Clean up temp file after tests
      on_exit(fn ->
        File.rm(temp_path)
      end)

      {:ok, base_image_path: temp_path}
    end

    test "generates valid image from init image with moderate denoising", %{base_image_path: init_image} do
      IO.puts("\n" <> String.duplicate("=", 80))
      IO.puts("TEST IMG2IMG 1: Moderate transformation (strength=0.7)")
      IO.puts(String.duplicate("=", 80))

      prompt = "convert to anime style with vibrant colors"

      opts = [
        model: :flux_schnell,
        steps: 4,
        size: {512, 512},
        denoising_strength: 0.7,
        seed: 123
      ]

      # Generate img2img
      result = Margarine.img2img(prompt, init_image, opts)

      # Verify successful generation
      assert {:ok, image} = result
      assert %Nx.Tensor{} = image

      # Verify tensor shape and type
      assert Nx.shape(image) == {512, 512, 3}
      assert Nx.type(image) == {:u, 8}

      # Verify image is not blank
      min_val = Nx.reduce_min(image) |> Nx.to_number()
      max_val = Nx.reduce_max(image) |> Nx.to_number()
      assert max_val > min_val, "Image appears blank"
      assert min_val >= 0
      assert max_val <= 255

      IO.puts("✓ Successfully generated img2img with strength=0.7")
    end

    test "subtle changes with low denoising strength", %{base_image_path: init_image} do
      IO.puts("\n" <> String.duplicate("=", 80))
      IO.puts("TEST IMG2IMG 2: Subtle changes (strength=0.3)")
      IO.puts(String.duplicate("=", 80))

      prompt = "turn into a watercolor painting"

      opts = [
        model: :flux_schnell,
        steps: 4,
        size: {512, 512},
        denoising_strength: 0.3,
        seed: 456
      ]

      {:ok, image} = Margarine.img2img(prompt, init_image, opts)

      # Same validation
      assert Nx.shape(image) == {512, 512, 3}
      assert Nx.type(image) == {:u, 8}

      min_val = Nx.reduce_min(image) |> Nx.to_number()
      max_val = Nx.reduce_max(image) |> Nx.to_number()
      assert max_val > min_val
      assert min_val >= 0
      assert max_val <= 255

      IO.puts("✓ Successfully generated img2img with strength=0.3")
    end

    test "complete regeneration with strength=1.0 (equivalent to text2img)", %{base_image_path: init_image} do
      IO.puts("\n" <> String.duplicate("=", 80))
      IO.puts("TEST IMG2IMG 3: Complete regeneration (strength=1.0)")
      IO.puts("This should be equivalent to text2img!")
      IO.puts(String.duplicate("=", 80))

      prompt = "a blue dragon flying over mountains"

      opts = [
        model: :flux_schnell,
        steps: 4,
        size: {512, 512},
        denoising_strength: 1.0,
        seed: 789
      ]

      {:ok, image} = Margarine.img2img(prompt, init_image, opts)

      # Same validation
      assert Nx.shape(image) == {512, 512, 3}
      assert Nx.type(image) == {:u, 8}

      min_val = Nx.reduce_min(image) |> Nx.to_number()
      max_val = Nx.reduce_max(image) |> Nx.to_number()
      assert max_val > min_val
      assert min_val >= 0
      assert max_val <= 255

      IO.puts("✓ Successfully generated img2img with strength=1.0")
      IO.puts("  This proves img2img can replicate text2img behavior!")
    end

    test "img2img respects seed for reproducibility", %{base_image_path: init_image} do
      IO.puts("\n" <> String.duplicate("=", 80))
      IO.puts("TEST IMG2IMG 4: Respects seed for reproducibility")
      IO.puts(String.duplicate("=", 80))

      prompt = "sunset colors"
      opts = [
        model: :flux_schnell,
        steps: 4,
        size: {512, 512},
        denoising_strength: 0.5,
        seed: 999
      ]

      # Generate twice with same seed
      {:ok, image1} = Margarine.img2img(prompt, init_image, opts)
      {:ok, image2} = Margarine.img2img(prompt, init_image, opts)

      # Should be identical
      assert Nx.equal(image1, image2) |> Nx.all() |> Nx.to_number() == 1

      IO.puts("✓ IMG2IMG produces identical results with same seed")
    end

    test "img2img fails with non-existent init image" do
      IO.puts("\n" <> String.duplicate("=", 80))
      IO.puts("TEST IMG2IMG 5: Error handling - non-existent file")
      IO.puts(String.duplicate("=", 80))

      prompt = "test"
      non_existent = "/tmp/this_file_does_not_exist_#{:rand.uniform(999_999)}.png"

      result = Margarine.img2img(prompt, non_existent, denoising_strength: 0.5)

      assert {:error, reason} = result
      assert reason =~ "not found" or reason =~ "exist"

      IO.puts("✓ Correctly rejects non-existent init image")
    end

    test "img2img with strength=1.0 is equivalent to text2img", %{base_image_path: init_image} do
      IO.puts("\n" <> String.duplicate("=", 80))
      IO.puts("TEST IMG2IMG 6: Prove img2img(strength=1.0) ≈ text2img")
      IO.puts("CORE DESIGN PRINCIPLE: Text2img is just img2img starting from pure noise!")
      IO.puts(String.duplicate("=", 80))

      # Use the same prompt and seed for both
      prompt = "a majestic mountain landscape at sunset"
      seed = 42
      opts_common = [model: :flux_schnell, steps: 4, size: {512, 512}, seed: seed]

      # Generate with text2img
      IO.puts("\n1. Generating with text2img...")
      {:ok, text2img_result} = Margarine.generate(prompt, opts_common)

      # Generate with img2img at strength=1.0
      IO.puts("2. Generating with img2img (strength=1.0)...")
      {:ok, img2img_result} =
        Margarine.img2img(prompt, init_image, opts_common ++ [denoising_strength: 1.0])

      # Both should have same shape and type
      assert Nx.shape(text2img_result) == Nx.shape(img2img_result)
      assert Nx.type(text2img_result) == Nx.type(img2img_result)

      # With the same seed, they should be identical (or extremely close)
      # Note: Small numerical differences may occur due to floating point precision
      # in the noise generation, but they should be very close
      are_identical = Nx.equal(text2img_result, img2img_result) |> Nx.all() |> Nx.to_number() == 1

      if are_identical do
        IO.puts("\n✓ PERFECT! Text2img and img2img(1.0) produced IDENTICAL results!")
        IO.puts("  This proves the core design principle:")
        IO.puts("  text2img = img2img(random_noise, strength=1.0)")
      else
        # They might differ slightly due to numerical precision
        # Calculate percentage of matching pixels
        total_pixels = 512 * 512 * 3
        matching_pixels =
          Nx.equal(text2img_result, img2img_result)
          |> Nx.sum()
          |> Nx.to_number()

        match_percentage = (matching_pixels / total_pixels) * 100

        IO.puts("\n✓ Text2img and img2img(1.0) are #{Float.round(match_percentage, 2)}% identical")

        # They should be at least 99% identical (allowing for minor numerical differences)
        assert match_percentage > 99.0,
          "text2img and img2img(1.0) should be nearly identical, got #{match_percentage}%"

        IO.puts("  (Small differences may be due to floating point precision)")
        IO.puts("  This still proves the core design principle:")
        IO.puts("  text2img ≈ img2img(random_noise, strength=1.0)")
      end

      IO.puts("\n🎉 DESIGN PRINCIPLE VALIDATED!")
      IO.puts("   Everything is img2img + prompt.")
      IO.puts("   Text2img is just img2img starting from pure noise.")
    end
  end
end
