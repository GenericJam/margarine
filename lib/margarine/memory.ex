defmodule Margarine.Memory do
  @moduledoc """
  Memory safety utilities for Margarine.

  Provides functions to check system memory availability before loading
  large models, preventing OOM crashes that could take down the entire machine.

  ## Model Memory Requirements

  FLUX models require significant memory:
  - FLUX Schnell: ~12-16GB
  - FLUX Dev: ~23-30GB

  These are estimates and actual usage may vary based on platform and configuration.
  """

  require Logger

  @type memory_info :: %{
          total: non_neg_integer(),
          available: non_neg_integer(),
          used: non_neg_integer(),
          percent_used: float()
        }

  # Model memory estimates in MB
  @model_memory_estimates %{
    flux_schnell: 14_000,
    # 14GB
    flux_dev: 26_000
    # 26GB
  }

  @doc """
  Gets current system memory information.

  Returns a map with total, available, used memory in bytes,
  and percentage used as a float.

  ## Examples

      {:ok, info} = Margarine.Memory.available_memory()
      IO.inspect(info)
      # => %{total: 68719476736, available: 12884901888, used: 55834574848, percent_used: 81.2}

  """
  @spec available_memory() :: {:ok, memory_info()} | {:error, String.t()}
  def available_memory do
    case :os.type() do
      {:unix, :darwin} ->
        get_macos_memory()

      {:unix, :linux} ->
        get_linux_memory()

      {:win32, _} ->
        get_windows_memory()

      other ->
        {:error, "Unsupported OS: #{inspect(other)}"}
    end
  end

  @doc """
  Checks if sufficient memory is available.

  ## Parameters

    * `required_mb` - Required memory in megabytes (must be positive integer)

  ## Examples

      :ok = Margarine.Memory.check_memory(100)

      {:error, reason} = Margarine.Memory.check_memory(999_999)

  """
  @spec check_memory(pos_integer()) :: :ok | {:error, String.t()}
  def check_memory(required_mb) when is_integer(required_mb) and required_mb > 0 do
    case available_memory() do
      {:ok, info} ->
        available_mb = bytes_to_mb(info.available)

        if available_mb >= required_mb do
          :ok
        else
          {:error,
           "Insufficient memory: required #{required_mb}MB, available #{available_mb}MB (#{format_bytes(info.available)})"}
        end

      {:error, reason} ->
        {:error, "Cannot check memory: #{reason}"}
    end
  end

  def check_memory(invalid) do
    {:error, "Invalid memory requirement: #{inspect(invalid)}. Must be a positive integer (MB)"}
  end

  @doc """
  Formats bytes as human-readable string.

  ## Examples

      iex> Margarine.Memory.format_bytes(1024)
      "1.0 KB"

      iex> Margarine.Memory.format_bytes(1_048_576)
      "1.0 MB"

      iex> Margarine.Memory.format_bytes(1_073_741_824)
      "1.0 GB"

  """
  @spec format_bytes(non_neg_integer()) :: String.t()
  def format_bytes(bytes) when bytes < 1024, do: "#{bytes} B"

  def format_bytes(bytes) when bytes < 1_048_576 do
    kb = bytes / 1024
    "#{Float.round(kb, 1)} KB"
  end

  def format_bytes(bytes) when bytes < 1_073_741_824 do
    mb = bytes / 1_048_576
    "#{Float.round(mb, 1)} MB"
  end

  def format_bytes(bytes) do
    gb = bytes / 1_073_741_824
    "#{Float.round(gb, 1)} GB"
  end

  @doc """
  Estimates memory requirements for a FLUX model.

  Returns estimated megabytes or error for unknown models.

  ## Examples

      iex> Margarine.Memory.estimate_flux_memory(:flux_schnell)
      14000

      iex> Margarine.Memory.estimate_flux_memory(:flux_dev)
      26000

  """
  @spec estimate_flux_memory(atom()) :: pos_integer() | {:error, String.t()}
  def estimate_flux_memory(model) when is_map_key(@model_memory_estimates, model) do
    @model_memory_estimates[model]
  end

  def estimate_flux_memory(model) do
    {:error, "Unknown model: #{inspect(model)}"}
  end

  @doc """
  Checks if sufficient memory is available for a FLUX model.

  Combines model memory estimation with memory checking.

  ## Examples

      :ok = Margarine.Memory.check_model_memory(:flux_schnell)

      {:error, reason} = Margarine.Memory.check_model_memory(:flux_dev)

  """
  @spec check_model_memory(atom()) :: :ok | {:error, String.t()}
  def check_model_memory(model) do
    case estimate_flux_memory(model) do
      {:error, _} = error ->
        error

      required_mb when is_integer(required_mb) ->
        case check_memory(required_mb) do
          :ok ->
            :ok

          {:error, reason} ->
            {:error, "Insufficient memory for FLUX #{model}: #{reason}"}
        end
    end
  end

  @doc """
  Checks available memory and returns detailed information.

  Alias for available_memory/0 with slightly different return format.

  ## Examples

      {:ok, %{available_bytes: 50_000_000_000, ...}} = Margarine.Memory.check_available()

  """
  @spec check_available() :: {:ok, map()} | {:error, String.t()}
  def check_available do
    case available_memory() do
      {:ok, info} ->
        {:ok, Map.put(info, :available_bytes, info.available)}

      {:error, _} = error ->
        error
    end
  end

  @doc """
  Converts megabytes to bytes.

  ## Examples

      iex> Margarine.Memory.mb_to_bytes(1)
      1048576

      iex> Margarine.Memory.mb_to_bytes(1024)
      1073741824

  """
  @spec mb_to_bytes(non_neg_integer()) :: non_neg_integer()
  def mb_to_bytes(mb), do: mb * 1_048_576

  @doc """
  Converts bytes to megabytes (rounded down).

  ## Examples

      iex> Margarine.Memory.bytes_to_mb(1_048_576)
      1

      iex> Margarine.Memory.bytes_to_mb(1_073_741_824)
      1024

  """
  @spec bytes_to_mb(non_neg_integer()) :: non_neg_integer()
  def bytes_to_mb(bytes), do: div(bytes, 1_048_576)

  # Private OS-specific implementations

  defp get_macos_memory do
    case System.cmd("vm_stat", []) do
      {output, 0} ->
        parse_macos_vm_stat(output)

      _ ->
        {:error, "Failed to execute vm_stat"}
    end
  rescue
    _ -> {:error, "vm_stat command not available"}
  end

  defp parse_macos_vm_stat(output) do
    lines = String.split(output, "\n")
    page_size = extract_page_size(lines)

    free = extract_pages(lines, "Pages free:")
    inactive = extract_pages(lines, "Pages inactive:")
    speculative = extract_pages(lines, "Pages speculative:")
    active = extract_pages(lines, "Pages active:")
    wired = extract_pages(lines, "Pages wired down:")

    total_pages = free + inactive + speculative + active + wired
    available_pages = free + inactive + speculative

    total_bytes = total_pages * page_size
    available_bytes = available_pages * page_size
    used_bytes = total_bytes - available_bytes
    percent_used = used_bytes / total_bytes * 100

    {:ok,
     %{
       total: total_bytes,
       available: available_bytes,
       used: used_bytes,
       percent_used: Float.round(percent_used, 1)
     }}
  rescue
    _ -> {:error, "Failed to parse vm_stat output"}
  end

  defp get_linux_memory do
    case File.read("/proc/meminfo") do
      {:ok, content} ->
        parse_linux_meminfo(content)

      _ ->
        {:error, "Failed to read /proc/meminfo"}
    end
  end

  defp parse_linux_meminfo(content) do
    total_kb = extract_kb(content, "MemTotal:")

    available_kb =
      case extract_kb(content, "MemAvailable:") do
        0 ->
          # Fallback for older kernels
          free = extract_kb(content, "MemFree:")
          buffers = extract_kb(content, "Buffers:")
          cached = extract_kb(content, "Cached:")
          free + buffers + cached

        available ->
          available
      end

    total_bytes = total_kb * 1024
    available_bytes = available_kb * 1024
    used_bytes = total_bytes - available_bytes
    percent_used = used_bytes / total_bytes * 100

    {:ok,
     %{
       total: total_bytes,
       available: available_bytes,
       used: used_bytes,
       percent_used: Float.round(percent_used, 1)
     }}
  rescue
    _ -> {:error, "Failed to parse /proc/meminfo"}
  end

  defp get_windows_memory do
    case System.cmd("wmic", ["OS", "get", "FreePhysicalMemory,TotalVisibleMemorySize"]) do
      {output, 0} ->
        parse_windows_wmic(output)

      _ ->
        {:error, "Failed to execute wmic"}
    end
  rescue
    _ -> {:error, "wmic command not available"}
  end

  defp parse_windows_wmic(output) do
    # Output format:
    # FreePhysicalMemory  TotalVisibleMemorySize
    # 1234567             8901234

    [_header | data] =
      output
      |> String.split("\n")
      |> Enum.map(&String.trim/1)
      |> Enum.reject(&(&1 == ""))

    case data do
      [memory_line | _] ->
        [free_kb_str, total_kb_str] =
          memory_line
          |> String.split(~r/\s+/, parts: 2)

        free_kb = String.to_integer(free_kb_str)
        total_kb = String.to_integer(total_kb_str)

        total_bytes = total_kb * 1024
        available_bytes = free_kb * 1024
        used_bytes = total_bytes - available_bytes
        percent_used = used_bytes / total_bytes * 100

        {:ok,
         %{
           total: total_bytes,
           available: available_bytes,
           used: used_bytes,
           percent_used: Float.round(percent_used, 1)
         }}

      _ ->
        {:error, "Failed to parse wmic output"}
    end
  rescue
    _ -> {:error, "Failed to parse wmic output"}
  end

  # Helpers for parsing

  defp extract_page_size(lines) do
    case Enum.find(lines, &String.contains?(&1, "page size")) do
      nil ->
        4096

      line ->
        case Regex.run(~r/page size of (\d+) bytes/, line) do
          [_, size] -> String.to_integer(size)
          _ -> 4096
        end
    end
  end

  defp extract_pages(lines, label) do
    case Enum.find(lines, &String.starts_with?(&1, label)) do
      nil ->
        0

      line ->
        line
        |> String.split(":")
        |> List.last()
        |> String.trim()
        |> String.trim_trailing(".")
        |> String.replace(~r/\s+/, "")
        |> String.to_integer()
    end
  rescue
    _ -> 0
  end

  defp extract_kb(content, label) do
    case Regex.run(~r/#{label}\s+(\d+) kB/, content) do
      [_, kb] -> String.to_integer(kb)
      nil -> 0
    end
  rescue
    _ -> 0
  end
end
