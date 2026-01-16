defmodule Margarine.Integration.MemoryLeakTest do
  use ExUnit.Case, async: false  # CRITICAL: Must run sequentially

  @moduledoc """
  Memory leak detection tests for Margarine.

  These tests verify that repeated image generations don't accumulate memory.
  They are designed to catch leaks in:
  - Python globals accumulation
  - Unreleased tensor references
  - Missing garbage collection
  - Pythonx memory issues

  **IMPORTANT:** Run with `mix test --only integration:memory_leak`
  """

  describe "memory leak detection" do
    @describetag :integration
    @describetag :memory_leak
    @describetag timeout: 900_000  # 15 minutes (generous for 10 generations)

    setup do
      on_exit(fn ->
        # Stop server after test
        server_name = :"Margarine.Python.PythonxServer.flux_schnell"

        case Process.whereis(server_name) do
          nil -> :ok
          pid -> GenServer.stop(pid, :normal, 5000)
        end

        # Force cleanup
        :erlang.garbage_collect()
        Process.sleep(1000)
      end)

      :ok
    end

    test "no memory leak over 10 small generations (512x512)" do
      IO.puts("\n" <> String.duplicate("=", 80))
      IO.puts("MEMORY LEAK TEST: 10 generations of 512x512 images")
      IO.puts(String.duplicate("=", 80))

      # Record initial memory
      {:ok, initial_info} = Margarine.Memory.available_memory()
      initial_mb = Margarine.Memory.bytes_to_mb(initial_info.available)
      initial_gb = initial_mb / 1024

      IO.puts("\n[Initial] Memory available: #{Float.round(initial_gb, 1)}GB (#{initial_mb}MB)")

      # Track memory usage across generations
      memory_samples = []

      # Generate 10 images
      for i <- 1..10 do
        IO.puts("\n--- Generation #{i}/10 ---")

        {:ok, _image} = Margarine.generate(
          "test image #{i}",
          model: :flux_schnell,
          steps: 4,
          size: {512, 512},
          seed: i
        )

        # Force GC after each generation
        :erlang.garbage_collect()
        Process.sleep(500)

        # Check memory after each generation
        {:ok, current_info} = Margarine.Memory.available_memory()
        current_mb = Margarine.Memory.bytes_to_mb(current_info.available)
        current_gb = current_mb / 1024

        leaked_mb = initial_mb - current_mb
        leaked_gb = leaked_mb / 1024

        memory_samples = memory_samples ++ [{i, current_mb, leaked_mb}]

        IO.puts("[Generation #{i}] Memory: #{Float.round(current_gb, 1)}GB available (leaked #{Float.round(leaked_gb, 1)}GB)")

        # Fail if we've leaked more than 3GB (very generous threshold)
        # This accounts for:
        # - Model staying in memory: ~12-14GB (expected)
        # - Normal Python overhead: ~500MB
        # - Leak allowance: ~1.5GB
        assert leaked_mb < 3000,
          """
          Memory leak detected!
          Leaked #{leaked_mb}MB after #{i} generations
          Initial: #{initial_mb}MB
          Current: #{current_mb}MB

          This suggests a memory leak in:
          - Python globals accumulation (fixed?)
          - Unreleased tensor references
          - Pythonx not freeing memory
          """
      end

      # Print summary
      IO.puts("\n" <> String.duplicate("=", 80))
      IO.puts("MEMORY LEAK TEST SUMMARY")
      IO.puts(String.duplicate("=", 80))

      Enum.each(memory_samples, fn {gen, available_mb, leaked_mb} ->
        IO.puts("Gen #{gen}: #{available_mb}MB available (leaked #{leaked_mb}MB)")
      end)

      # Check final memory
      {:ok, final_info} = Margarine.Memory.available_memory()
      final_mb = Margarine.Memory.bytes_to_mb(final_info.available)
      final_gb = final_mb / 1024
      total_leaked_mb = initial_mb - final_mb
      total_leaked_gb = total_leaked_mb / 1024

      IO.puts("\n[Final] Memory: #{Float.round(final_gb, 1)}GB available")
      IO.puts("[Final] Total leaked: #{Float.round(total_leaked_gb, 1)}GB (#{total_leaked_mb}MB)")

      if total_leaked_mb < 500 do
        IO.puts("\n✅ PASS: Memory leak is minimal (<500MB over 10 generations)")
      else
        IO.puts("\n⚠️  WARNING: Possible memory leak detected (#{total_leaked_mb}MB over 10 generations)")
      end

      # Final assertion: Total leak should be less than 3GB
      assert total_leaked_mb < 3000,
        "Total memory leak (#{total_leaked_mb}MB) exceeds threshold (3000MB)"
    end

    test "no memory leak over 3 large generations (1024x1024)" do
      IO.puts("\n" <> String.duplicate("=", 80))
      IO.puts("MEMORY LEAK TEST: 3 generations of 1024x1024 images")
      IO.puts(String.duplicate("=", 80))

      # Record initial memory
      {:ok, initial_info} = Margarine.Memory.available_memory()
      initial_mb = Margarine.Memory.bytes_to_mb(initial_info.available)
      initial_gb = initial_mb / 1024

      IO.puts("\n[Initial] Memory available: #{Float.round(initial_gb, 1)}GB (#{initial_mb}MB)")

      # Generate 3 large images (1024x1024 uses 4x more memory than 512x512)
      for i <- 1..3 do
        IO.puts("\n--- Large Generation #{i}/3 ---")

        {:ok, _image} = Margarine.generate(
          "large test image #{i}",
          model: :flux_schnell,
          steps: 4,
          size: {1024, 1024},
          seed: i * 100
        )

        # Force GC after each generation
        :erlang.garbage_collect()
        Process.sleep(1000)

        # Check memory
        {:ok, current_info} = Margarine.Memory.available_memory()
        current_mb = Margarine.Memory.bytes_to_mb(current_info.available)
        current_gb = current_mb / 1024
        leaked_mb = initial_mb - current_mb
        leaked_gb = leaked_mb / 1024

        IO.puts("[Generation #{i}] Memory: #{Float.round(current_gb, 1)}GB available (leaked #{Float.round(leaked_gb, 1)}GB)")

        # Fail if we've leaked more than 4GB
        # Large images need more temporary memory
        assert leaked_mb < 4000,
          "Memory leak detected: #{leaked_mb}MB after #{i} large generations"
      end

      # Final check
      {:ok, final_info} = Margarine.Memory.available_memory()
      final_mb = Margarine.Memory.bytes_to_mb(final_info.available)
      total_leaked_mb = initial_mb - final_mb

      IO.puts("\n[Final] Total leaked: #{Float.round(total_leaked_mb / 1024, 1)}GB (#{total_leaked_mb}MB)")

      assert total_leaked_mb < 4000,
        "Total memory leak (#{total_leaked_mb}MB) exceeds threshold for large images"
    end
  end
end
