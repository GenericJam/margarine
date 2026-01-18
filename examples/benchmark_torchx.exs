#!/usr/bin/env elixir

# Torchx Backend Benchmark
#
# Run with: `elixir examples/benchmark_torchx.exs`

defmodule BenchmarkHelpers do
  def format_time(ms) when ms < 1000, do: "#{ms}ms"
  def format_time(ms) when ms < 60_000, do: "#{Float.round(ms / 1000, 1)}s"
  def format_time(ms) do
    minutes = div(ms, 60_000)
    seconds = Float.round(rem(ms, 60_000) / 1000, 1)
    "#{minutes}m #{seconds}s"
  end
end

# Force CPU on both sides
System.put_env("MARGARINE_DEVICE", "mps")

Mix.install(
  [{:margarine, path: "."}, {:torchx, "~> 0.10"}],
  config: [
    nx: [default_backend: {Torchx.Backend, device: :mps}]
  ]
)

IO.puts("\n" <> String.duplicate("=", 80))
IO.puts("Torchx Backend Benchmark (MPS)")
IO.puts(String.duplicate("=", 80) <> "\n")

# Test configuration
config = [
  prompt: "a serene mountain landscape at sunset",
  model: :flux_schnell,
  steps: 4,
  size: {1024, 1024},
  seed: 42
]

IO.puts("Test Configuration:")
IO.puts("  Prompt: #{config[:prompt]}")
IO.puts("  Model: #{config[:model]}")
IO.puts("  Steps: #{config[:steps]}")
IO.puts("  Size: #{inspect(config[:size])}")
IO.puts("  Seed: #{config[:seed]}")
IO.puts("  Backend: Torchx (MPS)")
IO.puts("")

# Warmup
IO.puts("Warming up (loading model, not timed)...")
{:ok, _} = Margarine.generate(config[:prompt],
  model: config[:model],
  steps: 1,
  size: {512, 512},
  seed: config[:seed]
)
IO.puts("✓ Warmup complete\n")

# Benchmark
IO.puts("Running benchmark...")
start_time = System.monotonic_time(:millisecond)

result = Margarine.generate(config[:prompt],
  model: config[:model],
  steps: 4,
  size: config[:size],
  seed: config[:seed]
)

elapsed_ms = System.monotonic_time(:millisecond) - start_time

case result do
  {:ok, image} ->
    {h, w, c} = Nx.shape(image)
    Margarine.Image.save(image, "benchmark_torchx.png")

    IO.puts("✓ Generation complete!")
    IO.puts("")
    IO.puts("Results:")
    IO.puts("  Time: #{BenchmarkHelpers.format_time(elapsed_ms)}")
    IO.puts("  Image: #{h}x#{w}x#{c}")
    IO.puts("  Saved: benchmark_torchx.png")
    IO.puts("")

    # Save timing to file for comparison
    File.write!("benchmark_torchx_time.txt", "#{elapsed_ms}")
    IO.puts("Timing saved to: benchmark_torchx_time.txt")

  {:error, reason} ->
    IO.puts("✗ Generation failed: #{reason}")
    System.halt(1)
end

IO.puts("\n" <> String.duplicate("=", 80) <> "\n")
