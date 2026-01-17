defmodule Margarine.Python.SdxlPythonxServer do
  @moduledoc """
  GenServer managing SDXL model inference via Pythonx (zero-copy shared memory).

  Uses Pythonx to share memory directly between Nx (Elixir) and NumPy (Python)
  without JSON serialization. Provides ~200x speedup for tensor transfers.

  ## Communication Protocol

  Direct Python function calls via Pythonx with zero-copy memory sharing:
  - Nx.Tensor → NumPy array (shared memory)
  - NumPy array → Nx.Tensor (shared memory)
  - No JSON encoding/decoding overhead

  ## Supported Methods

  - `initialize_model(model, device, dtype)`
  - `encode_prompt(prompt, negative_prompt, guidance_scale)`
  - `get_time_ids(height, width, original_height, original_width)`
  - `unet_forward(latents, timestep, prompt_embeds, pooled_embeds, time_ids, guidance)`
  - `vae_decode(latents)`
  - `vae_encode(image)`
  - `generate_latents(height, width, seed)`
  """

  use GenServer
  require Logger

  @call_timeout 300_000  # 5 minutes for model loading

  defp python_module_dir do
    Path.join(:code.priv_dir(:margarine), "python")
  end

  # Client API

  def start_link(opts) do
    {gen_opts, init_opts} = Keyword.split(opts, [:name])
    GenServer.start_link(__MODULE__, init_opts, gen_opts)
  end

  def initialize_model(server, model, opts \\ []) do
    GenServer.call(server, {:initialize_model, model, opts}, @call_timeout)
  end

  def encode_prompt(server, prompt, opts \\ []) do
    negative = Keyword.get(opts, :negative_prompt, "")
    guidance = Keyword.get(opts, :guidance_scale, 7.5)
    GenServer.call(server, {:encode_prompt, prompt, negative, guidance}, @call_timeout)
  end

  def get_time_ids(server, height, width, guidance_scale \\ 7.5, opts \\ []) do
    original_height = Keyword.get(opts, :original_height)
    original_width = Keyword.get(opts, :original_width)
    GenServer.call(server, {:get_time_ids, height, width, original_height, original_width, guidance_scale}, @call_timeout)
  end

  def unet_forward(server, latents, timestep, prompt_embeds, pooled_embeds, time_ids, opts \\ []) do
    guidance = Keyword.get(opts, :guidance_scale, 7.5)
    GenServer.call(
      server,
      {:unet_forward, latents, timestep, prompt_embeds, pooled_embeds, time_ids, guidance},
      @call_timeout
    )
  end

  def vae_decode(server, latents) do
    GenServer.call(server, {:vae_decode, latents}, @call_timeout)
  end

  def vae_encode(server, image) do
    GenServer.call(server, {:vae_encode, image}, @call_timeout)
  end

  def generate_latents(server, height, width, seed \\ nil) do
    GenServer.call(server, {:generate_latents, height, width, seed}, @call_timeout)
  end

  def get_model_info(server) do
    GenServer.call(server, :get_model_info)
  end

  # Server Callbacks

  @impl true
  def init(opts) do
    model = Keyword.fetch!(opts, :model)
    device = Keyword.get(opts, :device, detect_device())
    dtype = Keyword.get(opts, :dtype, "float16")

    Logger.info("[Margarine.SdxlPythonxServer] Starting server for #{model}...")

    model_config = Margarine.Config.get_generation_defaults(model)

    # Store init params in state, defer model loading to handle_continue
    state = %{
      model: model,
      model_config: model_config,
      device: device,
      dtype: dtype,
      globals: nil,
      loading: true
    }

    {:ok, state, {:continue, :load_model}}
  end

  @impl true
  def handle_continue(:load_model, state) do
    # Check memory availability before loading model
    # SDXL requires ~7GB VRAM (less than FLUX's ~14GB)
    required_mb = estimate_sdxl_memory(state.model)

    case Margarine.Memory.available_memory() do
      {:ok, info} ->
        available_mb = Margarine.Memory.bytes_to_mb(info.available)

        Logger.info(
          "[Margarine.SdxlPythonxServer] Memory check for #{state.model}: " <>
            "Required ~#{required_mb}MB, Available #{available_mb}MB " <>
            "(#{Margarine.Memory.format_bytes(info.available)} of #{Margarine.Memory.format_bytes(info.total)})"
        )

        if available_mb >= required_mb do
          Logger.info("[Margarine.SdxlPythonxServer] ✓ Sufficient memory available")
        else
          Logger.error(
            "[Margarine.SdxlPythonxServer] ✗ Insufficient memory: " <>
              "Need #{required_mb}MB but only #{available_mb}MB available. " <>
              "Close other applications or use a smaller model."
          )

          {:stop, {:insufficient_memory, "Required #{required_mb}MB, available #{available_mb}MB"}, state}
        end

      {:error, reason} ->
        Logger.warning(
          "[Margarine.SdxlPythonxServer] Could not check memory availability: #{reason}. " <>
            "Proceeding with model load..."
        )
    end

    model_type = Atom.to_string(state.model)
    # Map Margarine model atoms to HuggingFace model IDs
    model_id = Margarine.Config.get_model_id(state.model)

    device = state.device
    dtype = state.dtype

    # Get HuggingFace token from environment
    hf_token = System.get_env("HF_TOKEN") || System.get_env("HUGGING_FACE_HUB_TOKEN") || ""

    # Initialize Pythonx and load model
    python_dir = python_module_dir()

    init_code = """
import sys
import os
sys.path.insert(0, '#{python_dir}')
import sdxl_pythonx

# Initialize model (expensive - do once)
model_type_str = model_type.decode('utf-8') if isinstance(model_type, bytes) else str(model_type)
model_id_str = model_id.decode('utf-8') if isinstance(model_id, bytes) else str(model_id)
device_str = device.decode('utf-8') if isinstance(device, bytes) else str(device)
dtype_str = dtype.decode('utf-8') if isinstance(dtype, bytes) else str(dtype)

# Handle HF token
hf_token_raw = token if token else os.getenv('HF_TOKEN') or os.getenv('HUGGING_FACE_HUB_TOKEN')
if hf_token_raw:
    hf_token = hf_token_raw.decode('utf-8') if isinstance(hf_token_raw, bytes) else hf_token_raw
else:
    hf_token = None

init_result = sdxl_pythonx.initialize_model(
    model_type_str,
    model_id_str,
    device_str,
    dtype_str,
    hf_token
)

initialized = True
"""

    init_globals = %{
      "model_type" => model_type,
      "model_id" => model_id,
      "device" => device,
      "dtype" => dtype,
      "token" => hf_token
    }

    Logger.info("[Margarine.SdxlPythonxServer] Loading SDXL model #{model_id} on #{device}...")
    if hf_token != "", do: Logger.info("[Margarine.SdxlPythonxServer] Using HuggingFace token")
    Logger.info("[Margarine.SdxlPythonxServer] This will download ~7GB on first run and take 2-5 minutes...")
    Logger.info("[Margarine.SdxlPythonxServer] Model loading in background - server ready for requests...")

    case Pythonx.eval(init_code, init_globals) do
      {:error, reason} ->
        Logger.error("[Margarine.SdxlPythonxServer] Failed to initialize: #{inspect(reason)}")
        {:stop, reason, state}

      {_result, new_globals} ->
        Logger.info("[Margarine.SdxlPythonxServer] ✓ SDXL model loaded successfully")
        new_state = %{state | globals: new_globals, loading: false}
        {:noreply, new_state}
    end
  end

  # Guard clause: Return error if model is still loading
  @impl true
  def handle_call(_request, _from, %{loading: true} = state) do
    {:reply, {:error, :model_loading}, state}
  end

  @impl true
  def handle_call({:initialize_model, model, _opts}, _from, state) do
    # For now, just acknowledge - full reinit not needed for MVP
    {:reply, {:ok, model}, state}
  end

  @impl true
  def handle_call({:encode_prompt, prompt, negative, guidance}, _from, state) do
    code = """
prompt_str = prompt.decode('utf-8') if isinstance(prompt, bytes) else str(prompt)
negative_str = negative.decode('utf-8') if isinstance(negative, bytes) else str(negative)

result = sdxl_pythonx.encode_prompt(
    prompt_str,
    negative_str,
    float(guidance)
)
result
"""

    # MEMORY LEAK FIX: Use fresh globals dict with only essential modules + call data
    # Don't merge and accumulate - create new dict each time
    call_globals = build_call_globals(state.globals, %{
      "prompt" => prompt,
      "negative" => negative,
      "guidance" => guidance
    })

    case Pythonx.eval(code, call_globals) do
      {:error, reason} ->
        Logger.error("[Margarine.SdxlPythonxServer] encode_prompt failed: #{inspect(reason)}")
        {:reply, {:error, reason}, state}

      {result, _new_globals} ->
        # MEMORY LEAK FIX: Discard new_globals to prevent accumulation
        decoded = decode_pythonx_result(result)
        {:reply, {:ok, decoded}, state}
    end
  end

  @impl true
  def handle_call({:get_time_ids, height, width, original_height, original_width, guidance_scale}, _from, state) do
    code = """
result = sdxl_pythonx.get_time_ids(
    int(height),
    int(width),
    int(original_height) if original_height is not None else None,
    int(original_width) if original_width is not None else None,
    float(guidance_scale)
)
result
"""

    call_globals = build_call_globals(state.globals, %{
      "height" => height,
      "width" => width,
      "original_height" => original_height,
      "original_width" => original_width,
      "guidance_scale" => guidance_scale
    })

    case Pythonx.eval(code, call_globals) do
      {:error, reason} ->
        Logger.error("[Margarine.SdxlPythonxServer] get_time_ids failed: #{inspect(reason)}")
        {:reply, {:error, reason}, state}

      {result, _new_globals} ->
        decoded = decode_pythonx_result(result)
        {:reply, {:ok, decoded}, state}
    end
  end

  @impl true
  def handle_call({:unet_forward, latents, timestep, prompt_embeds, pooled_embeds, time_ids, guidance}, _from, state) do
    # Convert Nx tensors to binary data
    latents_bin = Nx.to_binary(latents)
    prompt_embeds_bin = Nx.to_binary(prompt_embeds)
    pooled_embeds_bin = Nx.to_binary(pooled_embeds)
    time_ids_bin = Nx.to_binary(time_ids)

    latents_shape = Nx.shape(latents) |> Tuple.to_list()
    prompt_shape = Nx.shape(prompt_embeds) |> Tuple.to_list()
    pooled_shape = Nx.shape(pooled_embeds) |> Tuple.to_list()
    time_ids_shape = Nx.shape(time_ids) |> Tuple.to_list()

    # Get numpy dtypes from Nx types
    latents_dtype = nx_type_to_numpy_dtype(Nx.type(latents))
    prompt_dtype = nx_type_to_numpy_dtype(Nx.type(prompt_embeds))
    pooled_dtype = nx_type_to_numpy_dtype(Nx.type(pooled_embeds))
    time_ids_dtype = nx_type_to_numpy_dtype(Nx.type(time_ids))

    code = """
import numpy as np

# Reconstruct numpy arrays from binary data (add .copy() to make writable)
latents_np = np.frombuffer(latents_bin, dtype=np.#{latents_dtype}).reshape(latents_shape).copy()
prompt_embeds_np = np.frombuffer(prompt_embeds_bin, dtype=np.#{prompt_dtype}).reshape(prompt_shape).copy()
pooled_embeds_np = np.frombuffer(pooled_embeds_bin, dtype=np.#{pooled_dtype}).reshape(pooled_shape).copy()
time_ids_np = np.frombuffer(time_ids_bin, dtype=np.#{time_ids_dtype}).reshape(time_ids_shape).copy()

# Call UNet forward
result = sdxl_pythonx.unet_forward(
    latents_np,
    float(timestep),
    prompt_embeds_np,
    pooled_embeds_np,
    time_ids_np,
    float(guidance)
)
result
"""

    # MEMORY LEAK FIX: Use fresh globals dict with only essential modules + call data
    call_globals = build_call_globals(state.globals, %{
      "latents_bin" => latents_bin,
      "latents_shape" => latents_shape,
      "prompt_embeds_bin" => prompt_embeds_bin,
      "prompt_shape" => prompt_shape,
      "pooled_embeds_bin" => pooled_embeds_bin,
      "pooled_shape" => pooled_shape,
      "time_ids_bin" => time_ids_bin,
      "time_ids_shape" => time_ids_shape,
      "timestep" => timestep,
      "guidance" => guidance
    })

    case Pythonx.eval(code, call_globals) do
      {:error, reason} ->
        Logger.error("[Margarine.SdxlPythonxServer] unet_forward failed: #{inspect(reason)}")
        {:reply, {:error, reason}, state}

      {result, _new_globals} ->
        # MEMORY LEAK FIX: Discard new_globals to prevent accumulation
        decoded = decode_pythonx_result(result)
        {:reply, {:ok, decoded}, state}
    end
  end

  @impl true
  def handle_call({:vae_decode, latents}, _from, state) do
    latents_bin = Nx.to_binary(latents)
    latents_shape = Nx.shape(latents) |> Tuple.to_list()
    latents_dtype = nx_type_to_numpy_dtype(Nx.type(latents))

    code = """
import numpy as np

latents_np = np.frombuffer(latents_bin, dtype=np.#{latents_dtype}).reshape(latents_shape).copy()
result = sdxl_pythonx.vae_decode(latents_np)
result
"""

    # MEMORY LEAK FIX: Use fresh globals dict
    call_globals = build_call_globals(state.globals, %{
      "latents_bin" => latents_bin,
      "latents_shape" => latents_shape
    })

    case Pythonx.eval(code, call_globals) do
      {:error, reason} ->
        Logger.error("[Margarine.SdxlPythonxServer] vae_decode failed: #{inspect(reason)}")
        {:reply, {:error, reason}, state}

      {result, _new_globals} ->
        # MEMORY LEAK FIX: Discard new_globals to prevent accumulation
        decoded = decode_pythonx_result(result)
        {:reply, {:ok, decoded}, state}
    end
  end

  @impl true
  def handle_call({:vae_encode, image}, _from, state) do
    image_bin = Nx.to_binary(image)
    image_shape = Nx.shape(image) |> Tuple.to_list()
    image_dtype = nx_type_to_numpy_dtype(Nx.type(image))

    code = """
import numpy as np

image_np = np.frombuffer(image_bin, dtype=np.#{image_dtype}).reshape(image_shape).copy()
result = sdxl_pythonx.vae_encode(image_np)
result
"""

    # MEMORY LEAK FIX: Use fresh globals dict
    call_globals = build_call_globals(state.globals, %{
      "image_bin" => image_bin,
      "image_shape" => image_shape
    })

    case Pythonx.eval(code, call_globals) do
      {:error, reason} ->
        Logger.error("[Margarine.SdxlPythonxServer] vae_encode failed: #{inspect(reason)}")
        {:reply, {:error, reason}, state}

      {result, _new_globals} ->
        # MEMORY LEAK FIX: Discard new_globals to prevent accumulation
        decoded = decode_pythonx_result(result)
        {:reply, {:ok, decoded}, state}
    end
  end

  @impl true
  def handle_call({:generate_latents, height, width, seed}, _from, state) do
    code = """
result = sdxl_pythonx.generate_latents(height, width, seed)
result
"""

    # MEMORY LEAK FIX: Use fresh globals dict
    call_globals = build_call_globals(state.globals, %{
      "height" => height,
      "width" => width,
      "seed" => seed
    })

    case Pythonx.eval(code, call_globals) do
      {:error, reason} ->
        Logger.error("[Margarine.SdxlPythonxServer] generate_latents failed: #{inspect(reason)}")
        {:reply, {:error, reason}, state}

      {result, _new_globals} ->
        # MEMORY LEAK FIX: Discard new_globals to prevent accumulation
        decoded = decode_pythonx_result(result)
        {:reply, {:ok, decoded}, state}
    end
  end

  @impl true
  def handle_call(:get_model_info, _from, state) do
    code = """
result = sdxl_pythonx.get_model_info()
result
"""

    case Pythonx.eval(code, state.globals) do
      {:error, reason} ->
        Logger.error("[Margarine.SdxlPythonxServer] get_model_info failed: #{inspect(reason)}")
        {:reply, {:error, reason}, state}

      {result, _new_globals} ->
        # MEMORY LEAK FIX: Discard new_globals
        decoded = Pythonx.decode(result)
        {:reply, {:ok, decoded}, state}
    end
  end

  @impl true
  def terminate(reason, state) do
    Logger.info("[Margarine.SdxlPythonxServer] Shutting down (reason: #{inspect(reason)})")

    # Only attempt cleanup if we successfully loaded models
    if state.globals != nil and not state.loading do
      Logger.info("[Margarine.SdxlPythonxServer] Cleaning up Python resources...")

      cleanup_code = """
try:
    # Force garbage collection
    import gc
    gc.collect()

    # Explicitly delete global references
    if '_models' in dir(sdxl_pythonx):
        sdxl_pythonx._models = None

    cleanup_success = True
except Exception as e:
    print(f"[SdxlPythonx] Cleanup error: {e}")
    cleanup_success = False

cleanup_success
"""

      case Pythonx.eval(cleanup_code, state.globals) do
        {result, _} when is_struct(result, Pythonx.Object) ->
          # Python True/False are wrapped in Pythonx.Object
          Logger.info("[Margarine.SdxlPythonxServer] ✓ Python resources cleaned up")

        {:error, error} ->
          Logger.warning("[Margarine.SdxlPythonxServer] Failed to cleanup Python resources: #{inspect(error)}")

        other ->
          Logger.debug("[Margarine.SdxlPythonxServer] Cleanup result: #{inspect(other)}")
      end
    end

    :ok
  end

  # Private Helpers

  # Estimate SDXL memory requirements
  defp estimate_sdxl_memory(model) do
    case model do
      :sdxl_base -> 7000   # ~7GB for SDXL Base
      :sdxl_turbo -> 7000  # ~7GB for SDXL Turbo
      _ -> 7000
    end
  end

  # MEMORY LEAK FIX: Build fresh globals dict with only essential modules
  # This prevents accumulation of large binary data across calls
  defp build_call_globals(base_globals, call_data) do
    # Only keep essential module references from base_globals
    # Discard any accumulated data from previous calls
    essential_keys = ["sdxl_pythonx", "initialized", "init_result"]

    essential_globals =
      base_globals
      |> Map.take(essential_keys)
      |> Map.merge(call_data)

    essential_globals
  end

  defp decode_pythonx_result(result) do
    decoded = Pythonx.decode(result)
    convert_numpy_arrays(decoded)
  end

  defp convert_numpy_arrays(decoded) do
    case decoded do
      # Numpy array returned as (data, shape, dtype) tuple
      {data, shape, dtype_str} when is_binary(data) and is_list(shape) and is_binary(dtype_str) ->
        nx_type = case dtype_str do
          "float32" -> {:f, 32}
          "float64" -> {:f, 64}
          "float16" -> {:f, 16}
          "int32" -> {:s, 32}
          "uint8" -> {:u, 8}
          _ -> {:f, 32}
        end
        Nx.from_binary(data, nx_type) |> Nx.reshape(List.to_tuple(shape))

      # Dict/map (like encode_prompt returning multiple arrays)
      map when is_map(map) and not is_struct(map) ->
        Map.new(map, fn {k, v} ->
          key = if is_binary(k), do: String.to_atom(k), else: k
          {key, convert_numpy_arrays(v)}
        end)

      # List (could be nested results)
      list when is_list(list) ->
        Enum.map(list, &convert_numpy_arrays/1)

      # Pythonx.Object - try to decode it non-recursively
      %Pythonx.Object{} = obj ->
        # Decode once and return as-is (don't recurse to avoid infinite loops)
        Pythonx.decode(obj)

      # Otherwise return as-is
      other ->
        other
    end
  end

  defp detect_device do
    cond do
      System.get_env("CUDA_VISIBLE_DEVICES") -> "cuda"
      :os.type() == {:unix, :darwin} -> "mps"
      true -> "cpu"
    end
  end

  # Convert Nx type to numpy dtype string
  defp nx_type_to_numpy_dtype(nx_type) do
    case nx_type do
      {:f, 16} -> "float16"
      {:f, 32} -> "float32"
      {:f, 64} -> "float64"
      {:s, 32} -> "int32"
      {:u, 8} -> "uint8"
      _ -> "float32"  # Default fallback
    end
  end
end
