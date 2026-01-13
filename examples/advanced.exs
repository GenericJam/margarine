#!/usr/bin/env elixir

# Advanced Margarine Example
#
# This script demonstrates advanced features:
# - Custom models (FLUX Dev for higher quality)
# - Seed control for reproducibility
# - Custom image dimensions
# - Error handling and environment checks
# - Batch generation with different seeds
#
# Run with: `elixir examples/advanced.exs`

Mix.install([
  {:margarine, path: "."},
  {:emlx, "~> 0.1"}  # Change to {:exla, "~> 0.9"} for NVIDIA/AMD
])

# Configure Nx backend
Application.put_env(:nx, :default_backend, EMLX.Backend)
Application.put_env(:nx, :default_defn_options, compiler: EMLX)

defmodule MargarineAdvancedExample do
  @moduledoc false

  def run do
    IO.puts("\n🎨 Margarine - Advanced Example\n")

    # Check environment first
    check_environment()

    # Generate images with different configurations
    prompts = [
      {"a serene mountain landscape at sunset", :flux_schnell, 42},
      {"a cyberpunk city street at night", :flux_schnell, 123},
      {"a photorealistic portrait of a wise old wizard", :flux_dev, 999}
    ]

    IO.puts("\nGenerating #{length(prompts)} images...\n")

    results =
      prompts
      |> Enum.with_index(1)
      |> Enum.map(fn {{prompt, model, seed}, idx} ->
        generate_with_options(prompt, model, seed, idx, length(prompts))
      end)

    # Summary
    successes = Enum.count(results, fn {status, _} -> status == :ok end)
    failures = Enum.count(results, fn {status, _} -> status == :error end)

    IO.puts("\n" <> String.duplicate("=", 60))
    IO.puts("Summary: #{successes} succeeded, #{failures} failed")
    IO.puts(String.duplicate("=", 60) <> "\n")
  end

  defp check_environment do
    IO.puts("Checking environment...")

    case Margarine.check_environment() do
      %{pythonx_initialized: true, python_version: version} ->
        IO.puts("✓ Python #{version} ready\n")

      %{pythonx_initialized: false} ->
        IO.puts("⚠️  Python not initialized yet")
        IO.puts("First run will take 2-5 minutes to download dependencies.\n")
    end

    # Check available memory
    case Margarine.Memory.check_available() do
      {:ok, info} ->
        available_gb = Float.round(info.available_bytes / 1_073_741_824, 1)
        IO.puts("✓ Available RAM: #{available_gb} GB\n")

        if available_gb < 16 do
          IO.puts("⚠️  Warning: Less than 16GB RAM available.")
          IO.puts("Consider reducing image size or closing other applications.\n")
        end

      {:error, _reason} ->
        IO.puts("⚠️  Could not check available memory\n")
    end
  end

  defp generate_with_options(prompt, model, seed, idx, total) do
    IO.puts("#{idx}/#{total}: #{String.slice(prompt, 0..50)}...")
    IO.puts("  Model: #{model}, Seed: #{seed}")

    opts = [
      model: model,
      seed: seed,
      size: {1024, 1024},
      steps: model_steps(model),
      guidance_scale: model_guidance(model)
    ]

    start_time = System.monotonic_time(:millisecond)

    result = Margarine.generate(prompt, opts)

    elapsed_ms = System.monotonic_time(:millisecond) - start_time
    elapsed_sec = Float.round(elapsed_ms / 1000, 1)

    case result do
      {:ok, image} ->
        {h, w, c} = Nx.shape(image)

        # Save with descriptive filename
        filename = "output_#{idx}_#{model}_seed#{seed}.png"

        case Margarine.Image.save(image, filename) do
          :ok ->
            IO.puts("  ✓ Generated in #{elapsed_sec}s (#{h}x#{w}x#{c})")
            IO.puts("  ✓ Saved to: #{filename}\n")
            {:ok, filename}

          {:error, reason} ->
            IO.puts("  ✓ Generated in #{elapsed_sec}s")
            IO.puts("  ✗ Save failed: #{reason}\n")
            {:error, reason}
        end

      {:error, reason} ->
        IO.puts("  ✗ Failed: #{reason}")
        IO.puts("  Time elapsed: #{elapsed_sec}s\n")
        {:error, reason}
    end
  end

  defp model_steps(:flux_schnell), do: 4
  defp model_steps(:flux_dev), do: 28

  defp model_guidance(:flux_schnell), do: 0.0
  defp model_guidance(:flux_dev), do: 3.5
end

# Run the example
MargarineAdvancedExample.run()

IO.puts("✨ Done!\n")
