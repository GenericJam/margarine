#!/usr/bin/env elixir

# Backend Benchmark Script
#
# Compares EMLX vs Torchx performance for image generation
#
# Run with: `elixir examples/benchmark_backends.exs`

defmodule BackendBenchmark do
  @moduledoc false

  def run do
    IO.puts("\n" <> String.duplicate("=", 80))
    IO.puts("Margarine Backend Benchmark: EMLX vs Torchx")
    IO.puts(String.duplicate("=", 80) <> "\n")

    # Test parameters (same for both backends)
    test_config = [
      prompt: "a serene mountain landscape at sunset",
      model: :flux_schnell,
      steps: 4,
      size: {1024, 1024},
      seed: 42
    ]

    IO.puts("Test Configuration:")
    IO.puts("  Prompt: #{test_config[:prompt]}")
    IO.puts("  Model: #{test_config[:model]}")
    IO.puts("  Steps: #{test_config[:steps]}")
    IO.puts("  Size: #{inspect(test_config[:size])}")
    IO.puts("  Seed: #{test_config[:seed]}")
    IO.puts("")

    # Run benchmarks
    emlx_result = benchmark_emlx(test_config)
    cleanup_between_tests()
    torchx_result = benchmark_torchx(test_config)

    # Display results
    display_results(emlx_result, torchx_result)
  end

  defp benchmark_emlx(config) do
    IO.puts(String.duplicate("-", 80))
    IO.puts("Test 1: EMLX Backend (Apple Silicon MPS)")
    IO.puts(String.duplicate("-", 80))

    # Install EMLX
    IO.puts("\nInstalling EMLX backend...")

    Mix.install(
      [{:margarine, path: "."}],
      config: [
        nx: [default_backend: EMLX.Backend, default_defn_options: [compiler: EMLX]]
      ],
      force: true
    )

    # Ensure clean state
    Application.stop(:margarine)
    Application.ensure_all_started(:margarine)

    IO.puts("✓ EMLX backend installed and configured\n")

    # Warmup (model loading)
    IO.puts("Warming up (loading model, not timed)...")
    {:ok, _} = Margarine.generate(config[:prompt],
      model: config[:model],
      steps: 1,
      size: {512, 512},
      seed: config[:seed]
    )
    IO.puts("✓ Warmup complete\n")

    # Actual benchmark
    IO.puts("Running benchmark...")
    start_time = System.monotonic_time(:millisecond)

    result = Margarine.generate(config[:prompt],
      model: config[:model],
      steps: config[:steps],
      size: config[:size],
      seed: config[:seed]
    )

    elapsed_ms = System.monotonic_time(:millisecond) - start_time

    case result do
      {:ok, image} ->
        {h, w, c} = Nx.shape(image)
        Margarine.Image.save(image, "benchmark_emlx.png")

        IO.puts("✓ Generation complete!")
        IO.puts("  Time: #{format_time(elapsed_ms)}")
        IO.puts("  Image: #{h}x#{w}x#{c}")
        IO.puts("  Saved: benchmark_emlx.png\n")

        {:ok, elapsed_ms, image}

      {:error, reason} ->
        IO.puts("✗ Generation failed: #{reason}\n")
        {:error, reason}
    end
  end

  defp benchmark_torchx(config) do
    IO.puts(String.duplicate("-", 80))
    IO.puts("Test 2: Torchx Backend (CPU)")
    IO.puts(String.duplicate("-", 80))

    # Install Torchx
    IO.puts("\nInstalling Torchx backend...")

    # Force CPU on both sides
    System.put_env("MARGARINE_DEVICE", "cpu")

    Mix.install(
      [{:margarine, path: "."}, {:torchx, "~> 0.10"}],
      config: [
        nx: [default_backend: {Torchx.Backend, device: :cpu}]
      ],
      force: true
    )

    # Ensure clean state
    Application.stop(:margarine)
    Application.ensure_all_started(:margarine)

    IO.puts("✓ Torchx backend installed and configured")
    IO.puts("✓ Device set to CPU\n")

    # Warmup (model loading)
    IO.puts("Warming up (loading model, not timed)...")
    {:ok, _} = Margarine.generate(config[:prompt],
      model: config[:model],
      steps: 1,
      size: {512, 512},
      seed: config[:seed]
    )
    IO.puts("✓ Warmup complete\n")

    # Actual benchmark
    IO.puts("Running benchmark...")
    start_time = System.monotonic_time(:millisecond)

    result = Margarine.generate(config[:prompt],
      model: config[:model],
      steps: config[:steps],
      size: config[:size],
      seed: config[:seed]
    )

    elapsed_ms = System.monotonic_time(:millisecond) - start_time

    case result do
      {:ok, image} ->
        {h, w, c} = Nx.shape(image)
        Margarine.Image.save(image, "benchmark_torchx.png")

        IO.puts("✓ Generation complete!")
        IO.puts("  Time: #{format_time(elapsed_ms)}")
        IO.puts("  Image: #{h}x#{w}x#{c}")
        IO.puts("  Saved: benchmark_torchx.png\n")

        {:ok, elapsed_ms, image}

      {:error, reason} ->
        IO.puts("✗ Generation failed: #{reason}\n")
        {:error, reason}
    end
  end

  defp cleanup_between_tests do
    IO.puts("\nCleaning up between tests...")

    # Stop all applications
    Application.stop(:margarine)

    # Kill any running Python processes
    System.cmd("pkill", ["-f", "flux_pythonx"], stderr_to_stdout: true)

    # Give it a moment
    Process.sleep(2000)

    IO.puts("✓ Cleanup complete\n")
  end

  defp display_results({:ok, emlx_ms, _emlx_image}, {:ok, torchx_ms, _torchx_image}) do
    IO.puts("\n" <> String.duplicate("=", 80))
    IO.puts("BENCHMARK RESULTS")
    IO.puts(String.duplicate("=", 80) <> "\n")

    IO.puts("EMLX (Apple Silicon MPS):")
    IO.puts("  Time: #{format_time(emlx_ms)}")
    IO.puts("")

    IO.puts("Torchx (CPU):")
    IO.puts("  Time: #{format_time(torchx_ms)}")
    IO.puts("")

    # Calculate difference
    diff_ms = torchx_ms - emlx_ms
    diff_pct = Float.round((torchx_ms / emlx_ms - 1) * 100, 1)
    ratio = Float.round(torchx_ms / emlx_ms, 2)

    IO.puts("Comparison:")
    IO.puts("  Difference: #{format_time(abs(diff_ms))} #{if diff_ms > 0, do: "slower", else: "faster"}")
    IO.puts("  Percentage: #{abs(diff_pct)}% #{if diff_ms > 0, do: "slower", else: "faster"}")
    IO.puts("  Ratio: Torchx is #{ratio}x #{if ratio > 1, do: "slower", else: "faster"}")
    IO.puts("")

    # Winner
    winner = if emlx_ms < torchx_ms, do: "EMLX", else: "Torchx"
    IO.puts("Winner: #{winner} 🏆")
    IO.puts("")

    # Interpretation
    IO.puts("Interpretation:")
    cond do
      abs(diff_pct) < 10 ->
        IO.puts("  The backends have nearly identical performance (<10% difference).")
        IO.puts("  Both are suitable for production use on this hardware.")

      abs(diff_pct) < 30 ->
        IO.puts("  The backends have similar performance (#{abs(diff_pct)}% difference).")
        IO.puts("  Choice depends on other factors (compatibility, ecosystem, etc).")

      true ->
        IO.puts("  Significant performance difference (#{abs(diff_pct)}%).")
        IO.puts("  #{winner} is clearly faster for this workload.")
    end

    IO.puts("\n" <> String.duplicate("=", 80) <> "\n")
  end

  defp display_results({:error, emlx_error}, {:ok, torchx_ms, _}) do
    IO.puts("\n✗ EMLX failed: #{emlx_error}")
    IO.puts("✓ Torchx succeeded in #{format_time(torchx_ms)}")
    IO.puts("\nWinner: Torchx (EMLX failed) 🏆\n")
  end

  defp display_results({:ok, emlx_ms, _}, {:error, torchx_error}) do
    IO.puts("\n✓ EMLX succeeded in #{format_time(emlx_ms)}")
    IO.puts("✗ Torchx failed: #{torchx_error}")
    IO.puts("\nWinner: EMLX (Torchx failed) 🏆\n")
  end

  defp display_results({:error, emlx_error}, {:error, torchx_error}) do
    IO.puts("\n✗ EMLX failed: #{emlx_error}")
    IO.puts("✗ Torchx failed: #{torchx_error}")
    IO.puts("\nNo winner - both backends failed ❌\n")
  end

  defp format_time(ms) when ms < 1000 do
    "#{ms}ms"
  end

  defp format_time(ms) when ms < 60_000 do
    seconds = Float.round(ms / 1000, 1)
    "#{seconds}s"
  end

  defp format_time(ms) do
    minutes = div(ms, 60_000)
    seconds = Float.round(rem(ms, 60_000) / 1000, 1)
    "#{minutes}m #{seconds}s"
  end
end

# Run the benchmark
BackendBenchmark.run()

IO.puts("✨ Benchmark complete!\n")
IO.puts("Output files:")
IO.puts("  - benchmark_emlx.png")
IO.puts("  - benchmark_torchx.png")
IO.puts("\nCompare the images to verify they're identical (same seed was used).\n")
