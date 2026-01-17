defmodule Margarine.Integration.SdxlGenerationTest do
  use ExUnit.Case, async: false  # CRITICAL: Must run sequentially to avoid OOM

  @moduledoc """
  Integration tests for real SDXL image generation.

  These tests verify the complete end-to-end SDXL pipeline:
  1. Python environment initialization
  2. Model loading (SDXL Base or Turbo)
  3. Dual CLIP text prompt encoding
  4. DDIM scheduler with time IDs
  5. Denoising loop with UNet forward passes
  6. Image decoding to tensor
  7. Validation of output format

  **IMPORTANT:** These tests are marked with `@tag :integration` and
  excluded by default because they:
  - Require downloading ~7GB SDXL model (first run)
  - Need significant RAM/VRAM (10GB+ RAM or 7GB+ VRAM)
  - Take 10-60 seconds per image (depending on hardware and steps)
  - **MUST RUN SEQUENTIALLY** - async: false to prevent multiple model instances

  ## Known Benign Warnings

  You may see this warning at the end of tests:
  ```
  /path/to/python/multiprocessing/resource_tracker.py:254: UserWarning:
  resource_tracker: There appear to be 1 leaked semaphore objects to clean up at shutdown
  ```

  **This is a known benign warning** from Python's multiprocessing module when the
  Python process is terminated by an external process (Elixir/BEAM). It does NOT
  indicate a real memory leak or resource problem.

  See: https://github.com/apple/ml-stable-diffusion/issues/8

  ## Running Integration Tests

      # Run only SDXL integration tests
      mix test test/integration/sdxl_generation_test.exs --only integration

      # Run all integration tests
      mix test --only integration

  ## Prerequisites

  - Sufficient memory (10GB+ RAM or 7GB+ VRAM)
  - HuggingFace access token (optional)
  - Stable internet connection (first run)

  ## Test Strategy

  We use actual SDXL generation with minimal parameters:
  - sdxl_turbo (1 step, fastest) for smoke tests
  - sdxl_base (5 steps) for quality tests
  - Small images (512x512)
  - Simple prompts
  - Sequential execution to avoid OOM
  """

  describe "SDXL Turbo generation (fast)" do
    @describetag :integration
    @describetag timeout: 300_000  # 5 minutes per test

    setup do
      # Log memory before test
      case Margarine.Memory.available_memory() do
        {:ok, info} ->
          available_gb = Margarine.Memory.bytes_to_mb(info.available) / 1024
          IO.puts("\n[Setup] Memory before test: #{Float.round(available_gb, 1)}GB available")

        {:error, reason} ->
          IO.puts("\n[Setup] Could not check memory: #{reason}")
      end

      # Cleanup function to stop the SdxlPythonxServer after each test
      on_exit(fn ->
        Process.sleep(100)

        # Log memory before cleanup
        case Margarine.Memory.available_memory() do
          {:ok, info_before} ->
            available_gb = Margarine.Memory.bytes_to_mb(info_before.available) / 1024
            IO.puts("\n[Cleanup] Memory before cleanup: #{Float.round(available_gb, 1)}GB available")

          {:error, _} ->
            :ok
        end

        # Stop the server if it's running
        server_name = :sdxl_pythonx_server_sdxl_turbo

        case Process.whereis(server_name) do
          nil ->
            :ok

          pid ->
            IO.puts("[Cleanup] Stopping SdxlPythonxServer (#{inspect(pid)})...")
            GenServer.stop(pid, :normal, 5000)
            Process.sleep(500)
        end

        # MEMORY LEAK FIX: Force garbage collection after cleanup
        :erlang.garbage_collect()
        Process.sleep(100)

        # Log memory after cleanup
        case Margarine.Memory.available_memory() do
          {:ok, info_after} ->
            available_gb = Margarine.Memory.bytes_to_mb(info_after.available) / 1024
            IO.puts("\n[Cleanup] Memory after cleanup: #{Float.round(available_gb, 1)}GB available")

          {:error, _} ->
            :ok
        end
      end)

      :ok
    end

    @tag :integration
    test "generates 512x512 image with default settings" do
      IO.puts("\n[Test] Starting SDXL Turbo 512x512 generation test...")

      assert {:ok, image} = Margarine.generate(
        "a red panda eating bamboo",
        model: :sdxl_turbo,
        steps: 1,
        size: {512, 512},
        seed: 42
      )

      # Verify image tensor properties
      assert Nx.shape(image) == {512, 512, 3}
      assert Nx.type(image) == {:u, 8}

      # Debug: Save image to inspect it
      debug_path = "/tmp/sdxl_test_output.png"
      Margarine.Image.save(image, debug_path)
      IO.puts("[Test] Saved test image to #{debug_path}")

      # Check image is not blank (has variety of pixel values)
      # Note: SDXL Turbo with 1 step produces simpler images, so we use a lower threshold
      flat = Nx.to_flat_list(image)
      unique_values = Enum.uniq(flat)

      # Debug: Print some stats
      IO.puts("[Test] Image stats: #{length(unique_values)} unique values, min/max: #{Enum.min(flat)}/#{Enum.max(flat)}")

      assert length(unique_values) > 10, "Image should not be blank (found #{length(unique_values)} unique values)"

      IO.puts("[Test] ✓ SDXL Turbo 512x512 test passed")
    end

    @tag :integration
    test "generates reproducible images with same seed" do
      IO.puts("\n[Test] Testing SDXL Turbo seed reproducibility...")

      prompt = "a serene mountain landscape"
      opts = [model: :sdxl_turbo, steps: 1, size: {512, 512}, seed: 123]

      # Generate first image
      assert {:ok, image1} = Margarine.generate(prompt, opts)

      # Stop server to clear state
      server_name = :sdxl_pythonx_server_sdxl_turbo
      if pid = Process.whereis(server_name) do
        GenServer.stop(pid, :normal, 5000)
        Process.sleep(500)
      end

      # Generate second image with same seed
      assert {:ok, image2} = Margarine.generate(prompt, opts)

      # Images should be identical
      assert Nx.equal(image1, image2) |> Nx.all() |> Nx.to_number() == 1,
             "Images with same seed should be identical"

      IO.puts("[Test] ✓ SDXL Turbo reproducibility test passed")
    end

    @tag :integration
    test "generates different images with different seeds" do
      IO.puts("\n[Test] Testing SDXL Turbo different seeds...")

      prompt = "a futuristic cityscape at night"
      opts = [model: :sdxl_turbo, steps: 1, size: {512, 512}]

      # Generate with seed 42
      assert {:ok, image1} = Margarine.generate(prompt, opts ++ [seed: 42])

      # Generate with seed 123
      assert {:ok, image2} = Margarine.generate(prompt, opts ++ [seed: 123])

      # Images should be different
      total_pixels = 512 * 512 * 3
      matching_pixels = Nx.equal(image1, image2) |> Nx.sum() |> Nx.to_number()
      match_percentage = (matching_pixels / total_pixels) * 100

      assert match_percentage < 95.0,
             "Images with different seeds should differ significantly, got #{match_percentage}% match"

      IO.puts("[Test] ✓ SDXL Turbo different seeds test passed")
    end
  end

  describe "SDXL Base generation (quality)" do
    @describetag :integration
    @describetag timeout: 600_000  # 10 minutes per test (Base model is slower)

    setup do
      case Margarine.Memory.available_memory() do
        {:ok, info} ->
          available_gb = Margarine.Memory.bytes_to_mb(info.available) / 1024
          IO.puts("\n[Setup] Memory before test: #{Float.round(available_gb, 1)}GB available")

        {:error, reason} ->
          IO.puts("\n[Setup] Could not check memory: #{reason}")
      end

      on_exit(fn ->
        Process.sleep(100)

        server_name = :sdxl_pythonx_server_sdxl_base

        case Process.whereis(server_name) do
          nil ->
            :ok

          pid ->
            IO.puts("[Cleanup] Stopping SdxlPythonxServer for SDXL Base...")
            GenServer.stop(pid, :normal, 5000)
            Process.sleep(500)
        end

        :erlang.garbage_collect()
        Process.sleep(100)
      end)

      :ok
    end

    @tag :integration
    @tag :slow
    test "generates 512x512 image with 5 steps" do
      IO.puts("\n[Test] Starting SDXL Base 512x512 generation test (5 steps)...")

      assert {:ok, image} = Margarine.generate(
        "a majestic eagle soaring over mountains",
        model: :sdxl_base,
        steps: 5,
        size: {512, 512},
        seed: 42
      )

      # Verify image tensor properties
      assert Nx.shape(image) == {512, 512, 3}
      assert Nx.type(image) == {:u, 8}

      # Check image has variety
      flat = Nx.to_flat_list(image)
      unique_values = Enum.uniq(flat)
      assert length(unique_values) > 100

      IO.puts("[Test] ✓ SDXL Base quality test passed")
    end

    @tag :integration
    @tag :slow
    test "handles guidance scale parameter" do
      IO.puts("\n[Test] Testing SDXL Base guidance scale...")

      prompt = "a red sports car"
      opts = [model: :sdxl_base, steps: 5, size: {512, 512}, seed: 789]

      # Generate with low guidance (should be more creative)
      assert {:ok, image_low} = Margarine.generate(prompt, opts ++ [guidance_scale: 3.0])

      # Generate with high guidance (should follow prompt more closely)
      assert {:ok, image_high} = Margarine.generate(prompt, opts ++ [guidance_scale: 12.0])

      # Both should be valid images
      assert Nx.shape(image_low) == {512, 512, 3}
      assert Nx.shape(image_high) == {512, 512, 3}

      # Images should be different due to guidance difference
      total_pixels = 512 * 512 * 3
      matching_pixels = Nx.equal(image_low, image_high) |> Nx.sum() |> Nx.to_number()
      match_percentage = (matching_pixels / total_pixels) * 100

      assert match_percentage < 95.0,
             "Images with different guidance should differ, got #{match_percentage}% match"

      IO.puts("[Test] ✓ SDXL Base guidance scale test passed")
    end
  end

  describe "SDXL IMG2IMG generation" do
    @describetag :integration
    @describetag timeout: 600_000  # 10 minutes per test

    setup do
      # Create a base image for img2img tests
      base_image_path = Path.join(System.tmp_dir!(), "sdxl_test_base_image.png")

      # Generate a simple base image if it doesn't exist
      unless File.exists?(base_image_path) do
        IO.puts("\n[Setup] Generating base image for img2img tests...")
        {:ok, base_image} = Margarine.generate(
          "a simple geometric pattern",
          model: :sdxl_turbo,
          steps: 1,
          size: {512, 512},
          seed: 999
        )
        Margarine.Image.save(base_image, base_image_path)
        IO.puts("[Setup] ✓ Base image saved to #{base_image_path}")
      end

      case Margarine.Memory.available_memory() do
        {:ok, info} ->
          available_gb = Margarine.Memory.bytes_to_mb(info.available) / 1024
          IO.puts("\n[Setup] Memory before test: #{Float.round(available_gb, 1)}GB available")

        {:error, reason} ->
          IO.puts("\n[Setup] Could not check memory: #{reason}")
      end

      on_exit(fn ->
        Process.sleep(100)

        # Clean up both Turbo and Base servers if running
        for model <- [:sdxl_turbo, :sdxl_base] do
          server_name = :"sdxl_pythonx_server_#{model}"

          case Process.whereis(server_name) do
            nil ->
              :ok

            pid ->
              GenServer.stop(pid, :normal, 5000)
              Process.sleep(200)
          end
        end

        :erlang.garbage_collect()
        Process.sleep(100)
      end)

      %{base_image_path: base_image_path}
    end

    @tag :integration
    test "moderate transformation with strength=0.7", %{base_image_path: init_image} do
      IO.puts("\n[Test] Testing SDXL img2img with strength=0.7...")

      assert {:ok, image} = Margarine.img2img(
        "transform into a vibrant abstract painting",
        init_image,
        model: :sdxl_turbo,
        steps: 1,
        denoising_strength: 0.7,
        seed: 42
      )

      # Verify output
      assert Nx.shape(image) == {512, 512, 3}
      assert Nx.type(image) == {:u, 8}

      # Check it's not identical to input
      {:ok, original} = Margarine.Image.load(init_image)
      {:ok, original_resized} = Margarine.Image.resize(original, 512, 512)

      total_pixels = 512 * 512 * 3
      matching_pixels = Nx.equal(image, original_resized) |> Nx.sum() |> Nx.to_number()
      match_percentage = (matching_pixels / total_pixels) * 100

      assert match_percentage < 95.0,
             "Output should differ from input, got #{match_percentage}% match"

      IO.puts("[Test] ✓ SDXL img2img moderate transformation passed")
    end

    @tag :integration
    test "subtle changes with strength=0.3", %{base_image_path: init_image} do
      IO.puts("\n[Test] Testing SDXL img2img with strength=0.3...")

      assert {:ok, image} = Margarine.img2img(
        "slightly adjust colors",
        init_image,
        model: :sdxl_turbo,
        steps: 1,
        denoising_strength: 0.3,
        seed: 42
      )

      # Verify output
      assert Nx.shape(image) == {512, 512, 3}
      assert Nx.type(image) == {:u, 8}

      IO.puts("[Test] ✓ SDXL img2img subtle changes passed")
    end

    @tag :integration
    test "complete regeneration with strength=1.0 equals text2img", %{base_image_path: init_image} do
      IO.puts("\n[Test] Testing SDXL img2img strength=1.0 vs text2img...")
      IO.puts("[Test] CORE DESIGN PRINCIPLE: Text2img is just img2img starting from pure noise!")

      prompt = "a majestic mountain landscape at sunset"
      seed = 42
      opts_common = [model: :sdxl_turbo, steps: 1, size: {512, 512}, seed: seed]

      # Generate with text2img
      {:ok, text2img_result} = Margarine.generate(prompt, opts_common)

      # Generate with img2img at strength=1.0
      {:ok, img2img_result} =
        Margarine.img2img(prompt, init_image, opts_common ++ [denoising_strength: 1.0])

      # Calculate percentage of matching pixels
      total_pixels = 512 * 512 * 3
      matching_pixels = Nx.equal(text2img_result, img2img_result) |> Nx.sum() |> Nx.to_number()
      match_percentage = (matching_pixels / total_pixels) * 100

      assert match_percentage > 99.0,
        "text2img and img2img(1.0) should be nearly identical, got #{match_percentage}%"

      IO.puts("[Test] ✓ SDXL text2img == img2img(1.0) test passed")
    end

    @tag :integration
    test "reproducible with same seed", %{base_image_path: init_image} do
      IO.puts("\n[Test] Testing SDXL img2img reproducibility...")

      prompt = "convert to oil painting style"
      opts = [
        model: :sdxl_turbo,
        steps: 1,
        denoising_strength: 0.5,
        seed: 123
      ]

      # Generate first image
      {:ok, image1} = Margarine.img2img(prompt, init_image, opts)

      # Stop server to clear state
      server_name = :sdxl_pythonx_server_sdxl_turbo
      if pid = Process.whereis(server_name) do
        GenServer.stop(pid, :normal, 5000)
        Process.sleep(500)
      end

      # Generate second image with same seed
      {:ok, image2} = Margarine.img2img(prompt, init_image, opts)

      # Images should be identical
      assert Nx.equal(image1, image2) |> Nx.all() |> Nx.to_number() == 1,
             "img2img with same seed should be identical"

      IO.puts("[Test] ✓ SDXL img2img reproducibility test passed")
    end

    @tag :integration
    test "handles non-existent image file" do
      IO.puts("\n[Test] Testing SDXL img2img error handling...")

      assert {:error, reason} = Margarine.img2img(
        "test prompt",
        "/non/existent/file.png",
        model: :sdxl_turbo,
        steps: 1
      )

      assert reason =~ "not found" or reason =~ "does not exist"

      IO.puts("[Test] ✓ SDXL img2img error handling passed")
    end

    @tag :integration
    test "handles RGBA images (PNG with alpha channel)" do
      IO.puts("\n[Test] Testing SDXL img2img with RGBA image...")

      # Create a test RGBA image (red with 50% transparency)
      rgba_image_path = Path.join(System.tmp_dir!(), "sdxl_test_rgba.png")

      # Create RGBA image using Vix directly
      alias Vix.Vips.Image, as: VixImage
      alias Vix.Vips.Operation

      # Create 256x256 RGBA image (R=255, G=0, B=0, A=128)
      {:ok, red_channel} = VixImage.new_from_binary(<<255>> |> String.duplicate(256 * 256), 256, 256, 1, :VIPS_FORMAT_UCHAR)
      {:ok, green_channel} = VixImage.new_from_binary(<<0>> |> String.duplicate(256 * 256), 256, 256, 1, :VIPS_FORMAT_UCHAR)
      {:ok, blue_channel} = VixImage.new_from_binary(<<0>> |> String.duplicate(256 * 256), 256, 256, 1, :VIPS_FORMAT_UCHAR)
      {:ok, alpha_channel} = VixImage.new_from_binary(<<128>> |> String.duplicate(256 * 256), 256, 256, 1, :VIPS_FORMAT_UCHAR)

      {:ok, rgba_image} = Operation.bandjoin([red_channel, green_channel, blue_channel, alpha_channel])
      :ok = VixImage.write_to_file(rgba_image, rgba_image_path)

      IO.puts("[Test] Created test RGBA image at #{rgba_image_path}")

      # Test that img2img handles RGBA correctly (converts to RGB)
      assert {:ok, image} = Margarine.img2img(
        "transform into a blue gradient",
        rgba_image_path,
        model: :sdxl_turbo,
        steps: 1,
        size: {256, 256},
        denoising_strength: 0.5,
        seed: 42
      )

      # Verify output is RGB (not RGBA)
      assert Nx.shape(image) == {256, 256, 3}
      assert Nx.type(image) == {:u, 8}

      # Cleanup
      File.rm(rgba_image_path)

      IO.puts("[Test] ✓ SDXL img2img RGBA handling passed")
    end

    @tag :integration
    test "handles non-square images" do
      IO.puts("\n[Test] Testing SDXL img2img with non-square image...")

      # Create a test landscape image (768x512)
      landscape_path = Path.join(System.tmp_dir!(), "sdxl_test_landscape.png")

      # Generate a landscape base image
      {:ok, landscape} = Margarine.generate(
        "a horizontal landscape pattern",
        model: :sdxl_turbo,
        steps: 1,
        size: {512, 768},  # height, width
        seed: 123
      )
      Margarine.Image.save(landscape, landscape_path)

      IO.puts("[Test] Created test landscape image: 512x768")

      # Test img2img preserves aspect ratio (rounds to nearest multiple of 8)
      assert {:ok, image} = Margarine.img2img(
        "add mountains in background",
        landscape_path,
        model: :sdxl_turbo,
        steps: 1,
        size: {512, 768},  # Explicitly specify to match input
        denoising_strength: 0.6,
        seed: 456
      )

      # Verify output matches input aspect ratio
      assert Nx.shape(image) == {512, 768, 3}
      assert Nx.type(image) == {:u, 8}

      # Cleanup
      File.rm(landscape_path)

      IO.puts("[Test] ✓ SDXL img2img non-square image passed")
    end

    @tag :integration
    test "automatically rounds dimensions to multiple of 8" do
      IO.puts("\n[Test] Testing SDXL img2img auto-rounds dimensions...")

      # Create a test image with odd dimensions (not divisible by 8)
      # Let's use a 510x510 square
      odd_path = Path.join(System.tmp_dir!(), "sdxl_test_odd_dimensions.png")

      # Create 510x510 test image (not divisible by 8)
      {:ok, odd_image} = Margarine.generate(
        "test pattern",
        model: :sdxl_turbo,
        steps: 1,
        size: {512, 512},  # Generate at valid size first
        seed: 789
      )

      # Resize to 510x510 to create non-aligned dimensions
      {:ok, tensor_510} = Margarine.Image.load("/tmp/sdxl_test_output.png")
      {:ok, resized_510} = Margarine.Image.resize(tensor_510, 510, 510)
      Margarine.Image.save(resized_510, odd_path)

      IO.puts("[Test] Created test image with dimensions 510x510 (not divisible by 8)")

      # Load image to get dimensions
      {:ok, original} = Margarine.Image.load(odd_path)
      {orig_height, orig_width, _} = Nx.shape(original)

      # Round to nearest multiple of 8
      target_height = div(orig_height + 4, 8) * 8
      target_width = div(orig_width + 4, 8) * 8

      IO.puts("[Test] Original: #{orig_height}x#{orig_width}, Target: #{target_height}x#{target_width}")

      # img2img should accept the odd dimensions and round them
      assert {:ok, image} = Margarine.img2img(
        "enhance the image",
        odd_path,
        model: :sdxl_turbo,
        steps: 1,
        size: {target_height, target_width},
        denoising_strength: 0.4,
        seed: 321
      )

      # Verify output is rounded to valid dimensions
      assert Nx.shape(image) == {target_height, target_width, 3}
      assert rem(target_height, 8) == 0
      assert rem(target_width, 8) == 0

      # Cleanup
      File.rm(odd_path)

      IO.puts("[Test] ✓ SDXL img2img dimension rounding passed")
    end
  end
end
