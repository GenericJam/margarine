defmodule Margarine.Pipeline do
  @moduledoc """
  Pipeline coordinator for FLUX image generation.

  Orchestrates the complete generation workflow:
  1. Validate parameters
  2. Initialize scheduler
  3. Encode text prompt
  4. Generate initial noise
  5. Denoising loop
  6. Decode latents to image

  ## Example

      opts = [
        prompt: "a red panda eating bamboo",
        model: :flux_schnell,
        steps: 4,
        seed: 42
      ]

      {:ok, image} = Margarine.Pipeline.generate(opts)
  """

  alias Margarine.Config
  alias Margarine.Schedulers.FluxEuler
  alias Margarine.Python.PythonxServer

  require Logger

  @type pipeline_state :: %{
          prompt: String.t(),
          model: atom(),
          steps: pos_integer(),
          guidance_scale: float(),
          size: {pos_integer(), pos_integer()},
          seed: non_neg_integer() | nil,
          scheduler: map() | nil,
          latents: Nx.Tensor.t() | nil,
          prompt_embeds: Nx.Tensor.t() | nil,
          pooled_embeds: Nx.Tensor.t() | nil
        }

  @doc """
  Validate generation options.

  Checks all parameters and returns :ok or {:error, reason}.
  """
  @spec validate_opts(keyword()) :: :ok | {:error, String.t()}
  def validate_opts(opts) do
    prompt = Keyword.get(opts, :prompt)
    model = Keyword.get(opts, :model, :flux_schnell)
    steps = Keyword.get(opts, :steps)
    guidance_scale = Keyword.get(opts, :guidance_scale)
    size = Keyword.get(opts, :size, {1024, 1024})
    seed = Keyword.get(opts, :seed)

    # Chain validations using with
    with :ok <- validate_prompt(prompt),
         :ok <- validate_model(model),
         :ok <- validate_steps(steps, model),
         :ok <- validate_guidance_scale(guidance_scale, model),
         :ok <- validate_size(size) do
      validate_seed(seed)
    end
  end

  @doc """
  Prepare pipeline state from options.

  Validates options and initializes the pipeline state.
  """
  @spec prepare(keyword()) :: {:ok, pipeline_state()} | {:error, String.t()}
  def prepare(opts) do
    case validate_opts(opts) do
      :ok ->
        # Get defaults from Config
        model = Keyword.get(opts, :model, :flux_schnell)
        defaults = Config.get_generation_defaults(model)

        state = %{
          prompt: Keyword.fetch!(opts, :prompt),
          model: model,
          steps: Keyword.get(opts, :steps, defaults.steps),
          guidance_scale: Keyword.get(opts, :guidance_scale, defaults.guidance_scale),
          size: Keyword.get(opts, :size, defaults.size),
          seed: Keyword.get(opts, :seed),
          scheduler: nil,
          latents: nil,
          prompt_embeds: nil,
          pooled_embeds: nil
        }

        {:ok, state}

      error ->
        error
    end
  end

  @doc """
  Execute the complete FLUX generation pipeline.

  Takes a prepared pipeline state and generates an image.

  ## Steps:
  1. Start/get PythonxServer GenServer
  2. Encode text prompt to embeddings
  3. Initialize scheduler with timesteps
  4. Generate initial latents (noise)
  5. Denoising loop (transformer forward passes)
  6. Decode latents to image

  Returns `{:ok, image_tensor}` where image is {H, W, 3} uint8 RGB tensor.
  """
  @spec generate(pipeline_state()) :: {:ok, Nx.Tensor.t()} | {:error, String.t()}
  def generate(state) do
    Logger.info("[Margarine.Pipeline] Starting generation: #{state.prompt}")
    Logger.info("[Margarine.Pipeline] Model: #{state.model}, Steps: #{state.steps}, Size: #{inspect(state.size)}")

    with {:ok, server} <- get_or_start_server(state.model),
         {:ok, embeds} <- encode_prompt(server, state),
         {:ok, scheduler} <- initialize_scheduler(state),
         {:ok, latents} <- generate_initial_latents(server, state),
         {:ok, denoised} <- denoising_loop(server, state, scheduler, latents, embeds),
         {:ok, image} <- decode_to_image(server, denoised) do
      Logger.info("[Margarine.Pipeline] ✓ Generation complete")
      {:ok, image}
    end
  end

  # Private pipeline steps

  defp get_or_start_server(model) do
    server_name = server_name_for_model(model)

    case Process.whereis(server_name) do
      nil ->
        Logger.info("[Margarine.Pipeline] Starting PythonxServer for #{model}...")
        opts = [name: server_name, model: model]

        case PythonxServer.start_link(opts) do
          {:ok, pid} ->
            Logger.info("[Margarine.Pipeline] ✓ PythonxServer started: #{inspect(pid)}")
            {:ok, server_name}

          {:error, {:already_started, _pid}} ->
            {:ok, server_name}

          {:error, reason} ->
            {:error, "Failed to start Python server: #{inspect(reason)}"}
        end

      _pid ->
        {:ok, server_name}
    end
  end

  defp encode_prompt(server, state) do
    Logger.info("[Margarine.Pipeline] Encoding prompt...")

    case PythonxServer.encode_prompt(server, state.prompt,
           guidance_scale: state.guidance_scale
         ) do
      {:ok, result} ->
        Logger.info("[Margarine.Pipeline] ✓ Prompt encoded")
        {:ok, result}

      {:error, reason} ->
        {:error, "Prompt encoding failed: #{inspect(reason)}"}
    end
  end

  defp initialize_scheduler(state) do
    Logger.info("[Margarine.Pipeline] Initializing scheduler...")
    scheduler = FluxEuler.new() |> FluxEuler.set_timesteps(state.steps)
    Logger.info("[Margarine.Pipeline] ✓ Scheduler initialized with #{state.steps} steps")
    {:ok, scheduler}
  end

  defp generate_initial_latents(server, state) do
    Logger.info("[Margarine.Pipeline] Generating initial latents...")
    {height, width} = state.size

    case PythonxServer.generate_latents(server, height, width, state.seed) do
      {:ok, latents} ->
        Logger.info("[Margarine.Pipeline] ✓ Initial latents generated")
        {:ok, latents}

      {:error, reason} ->
        {:error, "Latent generation failed: #{inspect(reason)}"}
    end
  end

  defp denoising_loop(server, state, scheduler, latents, embeds) do
    Logger.info("[Margarine.Pipeline] Starting denoising loop (#{state.steps} steps)...")

    timesteps = scheduler.timesteps |> Nx.to_flat_list()
    num_steps = length(timesteps)

    result =
      Enum.reduce_while(Enum.with_index(timesteps), latents, fn {timestep, idx}, current_latents ->
        Logger.debug("[Margarine.Pipeline] Step #{idx + 1}/#{num_steps}, timestep: #{timestep}")

        case PythonxServer.transformer_forward(
               server,
               current_latents,
               timestep,
               embeds.prompt_embeds,
               embeds.pooled_embeds,
               guidance_scale: state.guidance_scale
             ) do
          {:ok, model_output} ->
            # Apply scheduler step
            next_latents =
              FluxEuler.step(scheduler, model_output, idx, current_latents)

            {:cont, next_latents}

          {:error, reason} ->
            {:halt, {:error, "Denoising step #{idx} failed: #{inspect(reason)}"}}
        end
      end)

    case result do
      {:error, _} = error ->
        error

      final_latents ->
        Logger.info("[Margarine.Pipeline] ✓ Denoising complete")
        {:ok, final_latents}
    end
  end

  defp decode_to_image(server, latents) do
    Logger.info("[Margarine.Pipeline] Decoding latents to image...")

    case PythonxServer.vae_decode(server, latents) do
      {:ok, image} ->
        Logger.info("[Margarine.Pipeline] ✓ Image decoded")
        {:ok, image}

      {:error, reason} ->
        {:error, "VAE decode failed: #{inspect(reason)}"}
    end
  end

  defp server_name_for_model(model) do
    String.to_atom("margarine_pythonx_#{model}")
  end

  # Private validation helpers

  defp validate_prompt(nil), do: {:error, "prompt is required"}
  defp validate_prompt(prompt) when is_binary(prompt) and byte_size(prompt) > 0, do: :ok
  defp validate_prompt(_), do: {:error, "prompt must be a non-empty string"}

  defp validate_model(model) when model in [:flux_schnell, :flux_dev], do: :ok
  defp validate_model(model), do: {:error, "model must be :flux_schnell or :flux_dev, got: #{inspect(model)}"}

  defp validate_steps(nil, model) do
    # Will use defaults from Config
    _ = model
    :ok
  end

  defp validate_steps(steps, _model) when is_integer(steps) and steps > 0, do: :ok

  defp validate_steps(steps, _model),
    do: {:error, "steps must be a positive integer, got: #{inspect(steps)}"}

  defp validate_guidance_scale(nil, _model), do: :ok

  defp validate_guidance_scale(scale, _model) when is_number(scale) and scale >= 0.0,
    do: :ok

  defp validate_guidance_scale(scale, _model),
    do: {:error, "guidance_scale must be >= 0.0, got: #{inspect(scale)}"}

  defp validate_size({w, h}) when is_integer(w) and is_integer(h) and w > 0 and h > 0 do
    # Check divisible by 8 (FLUX requirement)
    case {rem(w, 8), rem(h, 8)} do
      {0, 0} -> :ok
      _ -> {:error, "size dimensions must be divisible by 8, got: {#{w}, #{h}}"}
    end
  end

  defp validate_size(size), do: {:error, "size must be {width, height} tuple, got: #{inspect(size)}"}

  defp validate_seed(nil), do: :ok
  defp validate_seed(seed) when is_integer(seed) and seed >= 0, do: :ok
  defp validate_seed(seed), do: {:error, "seed must be a non-negative integer, got: #{inspect(seed)}"}
end
