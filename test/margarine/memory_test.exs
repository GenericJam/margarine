defmodule Margarine.MemoryTest do
  use ExUnit.Case, async: true

  @moduledoc """
  Tests for Margarine.Memory module.

  NOTE ON COVERAGE: The Memory module contains OS-specific code paths
  (macOS, Linux, Windows) that cannot all be tested in a single CI run.
  Coverage will vary by platform:
  - macOS: Tests get_macos_memory path
  - Linux: Tests get_linux_memory path
  - Windows: Tests get_windows_memory path

  This is acceptable per TDD guidelines for interfacing with external
  systems that can't be fully tested.
  """

  alias Margarine.Memory

  describe "available_memory/0" do
    test "returns memory information as a map" do
      assert {:ok, info} = Memory.available_memory()

      # Should have total and available memory in bytes
      assert is_integer(info.total)
      assert is_integer(info.available)
      assert info.total > 0
      assert info.available > 0
      assert info.available <= info.total

      # Should have used memory
      assert is_integer(info.used)
      assert info.used >= 0
      assert info.used <= info.total

      # Should have percentage used
      assert is_float(info.percent_used)
      assert info.percent_used >= 0.0
      assert info.percent_used <= 100.0
    end

    test "can be called multiple times" do
      assert {:ok, _info1} = Memory.available_memory()
      assert {:ok, _info2} = Memory.available_memory()
    end
  end

  describe "check_memory/1" do
    test "succeeds when enough memory is available" do
      # Request 1MB (should always be available on test machines)
      required_mb = 1
      assert :ok = Memory.check_memory(required_mb)
    end

    test "fails when insufficient memory" do
      # Request an absurdly large amount
      required_mb = 999_999_999

      assert {:error, reason} = Memory.check_memory(required_mb)
      assert reason =~ "Insufficient memory"
      assert reason =~ "required"
      assert reason =~ "available"
    end

    test "accepts memory requirement in MB as integer" do
      assert :ok = Memory.check_memory(1)
    end

    test "rejects invalid memory requirements" do
      assert {:error, reason} = Memory.check_memory(-1)
      assert reason =~ "Invalid"

      assert {:error, reason} = Memory.check_memory(0)
      assert reason =~ "Invalid"

      assert {:error, reason} = Memory.check_memory("100")
      assert reason =~ "Invalid"
    end
  end

  describe "format_bytes/1" do
    test "formats bytes as human-readable string" do
      assert Memory.format_bytes(0) == "0 B"
      assert Memory.format_bytes(500) == "500 B"
      assert Memory.format_bytes(1024) == "1.0 KB"
      assert Memory.format_bytes(1536) == "1.5 KB"
      assert Memory.format_bytes(1_048_576) == "1.0 MB"
      assert Memory.format_bytes(1_073_741_824) == "1.0 GB"
      assert Memory.format_bytes(5_368_709_120) == "5.0 GB"
    end

    test "handles large values" do
      # 64 GB
      assert Memory.format_bytes(68_719_476_736) == "64.0 GB"
    end

    test "rounds to 1 decimal place" do
      # 1.234 GB
      assert Memory.format_bytes(1_324_997_410) =~ ~r/1\.\d GB/
    end
  end

  describe "estimate_flux_memory/1" do
    test "estimates FLUX Schnell memory requirements" do
      memory_mb = Memory.estimate_flux_memory(:flux_schnell)

      # FLUX Schnell should be around 12-16GB
      assert is_integer(memory_mb)
      assert memory_mb >= 10_000
      assert memory_mb <= 20_000
    end

    test "estimates FLUX Dev memory requirements" do
      memory_mb = Memory.estimate_flux_memory(:flux_dev)

      # FLUX Dev is larger, around 23-30GB
      assert is_integer(memory_mb)
      assert memory_mb >= 20_000
      assert memory_mb <= 35_000
    end

    test "returns error for unknown model" do
      assert {:error, reason} = Memory.estimate_flux_memory(:unknown_model)
      assert reason =~ "Unknown model"
    end
  end

  describe "check_model_memory/1" do
    test "checks if enough memory for FLUX Schnell" do
      # This will pass or fail depending on actual system memory
      result = Memory.check_model_memory(:flux_schnell)

      case result do
        :ok ->
          # System has enough memory (>12GB available)
          {:ok, info} = Memory.available_memory()
          assert info.available > 12_000_000_000

        {:error, reason} ->
          # System doesn't have enough memory
          assert reason =~ "Insufficient memory"
          assert reason =~ "FLUX"
      end
    end

    test "returns descriptive error for insufficient memory" do
      # FLUX Dev requires more memory, more likely to fail
      case Memory.check_model_memory(:flux_dev) do
        :ok ->
          # System has lots of memory
          {:ok, info} = Memory.available_memory()
          assert info.available > 23_000_000_000

        {:error, reason} ->
          # Error should be descriptive
          assert reason =~ "Insufficient memory"
          assert reason =~ "FLUX"
          assert reason =~ "required"
          assert reason =~ "available"
      end
    end
  end

  describe "mb_to_bytes/1 and bytes_to_mb/1" do
    test "converts MB to bytes" do
      assert Memory.mb_to_bytes(1) == 1_048_576
      assert Memory.mb_to_bytes(1024) == 1_073_741_824
      assert Memory.mb_to_bytes(0) == 0
    end

    test "converts bytes to MB" do
      assert Memory.bytes_to_mb(1_048_576) == 1
      assert Memory.bytes_to_mb(1_073_741_824) == 1024
      assert Memory.bytes_to_mb(0) == 0
    end

    test "round-trip conversion" do
      mb = 1337
      assert mb == Memory.bytes_to_mb(Memory.mb_to_bytes(mb))
    end
  end
end
