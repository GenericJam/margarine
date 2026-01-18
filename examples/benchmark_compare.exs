#!/usr/bin/env elixir

# Compare Benchmark Results
#
# Run after benchmark_emlx.exs and benchmark_torchx.exs
#
# Run with: `elixir examples/benchmark_compare.exs`

defmodule BenchmarkHelpers do
  def format_time(ms) when ms < 1000, do: "#{ms}ms"
  def format_time(ms) when ms < 60_000, do: "#{Float.round(ms / 1000, 1)}s"
  def format_time(ms) do
    minutes = div(ms, 60_000)
    seconds = Float.round(rem(ms, 60_000) / 1000, 1)
    "#{minutes}m #{seconds}s"
  end
end

IO.puts("\n" <> String.duplicate("=", 80))
IO.puts("Backend Benchmark Comparison")
IO.puts(String.duplicate("=", 80) <> "\n")

# Read timing files
emlx_ms = case File.read("benchmark_emlx_time.txt") do
  {:ok, content} -> String.to_integer(String.trim(content))
  {:error, _} ->
    IO.puts("Error: benchmark_emlx_time.txt not found")
    IO.puts("Please run: elixir examples/benchmark_emlx.exs")
    System.halt(1)
end

torchx_ms = case File.read("benchmark_torchx_time.txt") do
  {:ok, content} -> String.to_integer(String.trim(content))
  {:error, _} ->
    IO.puts("Error: benchmark_torchx_time.txt not found")
    IO.puts("Please run: elixir examples/benchmark_torchx.exs")
    System.halt(1)
end

# Display results
IO.puts("EMLX (Apple Silicon MPS):")
IO.puts("  Time: #{BenchmarkHelpers.format_time(emlx_ms)}")
IO.puts("")

IO.puts("Torchx (MPS):")
IO.puts("  Time: #{BenchmarkHelpers.format_time(torchx_ms)}")
IO.puts("")

# Calculate comparison
diff_ms = torchx_ms - emlx_ms
diff_pct = Float.round((torchx_ms / emlx_ms - 1) * 100, 1)
ratio = Float.round(torchx_ms / emlx_ms, 2)

IO.puts("Comparison:")
IO.puts("  Difference: #{BenchmarkHelpers.format_time(abs(diff_ms))} #{if diff_ms > 0, do: "slower", else: "faster"}")
IO.puts("  Percentage: #{abs(diff_pct)}% #{if diff_ms > 0, do: "slower", else: "faster"}")
IO.puts("  Ratio: Torchx is #{ratio}x #{if ratio > 1, do: "slower", else: "faster"} than EMLX")
IO.puts("")

# Winner
winner = if emlx_ms < torchx_ms, do: "EMLX", else: "Torchx"
IO.puts("Winner: #{winner} 🏆")
IO.puts("")

# Interpretation
IO.puts("Interpretation:")
cond do
  abs(diff_pct) < 10 ->
    IO.puts("  ✓ The backends have nearly identical performance (<10% difference).")
    IO.puts("  ✓ Both are suitable for production use on this hardware.")
    IO.puts("  → Backend choice can be based on other factors (compatibility, ecosystem).")

  abs(diff_pct) < 30 ->
    IO.puts("  ✓ The backends have similar performance (#{abs(diff_pct)}% difference).")
    IO.puts("  → Choice depends on other factors (compatibility, ecosystem, etc).")

  diff_pct > 0 ->
    IO.puts("  ⚠ Torchx is significantly slower (#{diff_pct}%).")
    IO.puts("  → EMLX is clearly better for this hardware (Apple Silicon).")

  true ->
    IO.puts("  ⚠ EMLX is significantly slower (#{abs(diff_pct)}%).")
    IO.puts("  → Torchx is unexpectedly faster - this warrants investigation.")
end

IO.puts("")
IO.puts("Output files:")
IO.puts("  - benchmark_emlx.png")
IO.puts("  - benchmark_torchx.png")
IO.puts("")
IO.puts("Visual comparison:")
IO.puts("  Both images should be identical (same seed: 42)")
IO.puts("  If they differ, this indicates backend differences in random number generation.")

IO.puts("\n" <> String.duplicate("=", 80) <> "\n")
