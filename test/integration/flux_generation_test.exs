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
end
