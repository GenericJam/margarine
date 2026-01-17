defmodule Margarine.Pipelines.Sdxl do
  @moduledoc """
  Pipeline coordinator for SDXL image generation.

  Orchestrates the complete generation workflow:
  1. Validate parameters
  2. Initialize DDIM scheduler
  3. Encode text prompt (dual CLIP encoders)
  4. Generate initial noise
  5. Denoising loop (UNet forward passes)
  6. Decode latents to image

  ## Example

      opts = [
        prompt: "a red panda eating bamboo",
        model: :sdxl_base,
        steps: 20,
        seed: 42
      ]

      {:ok, image} = Margarine.Pipelines.Sdxl.generate(opts)
  """

  alias Margarine.Config
  alias Margarine.Schedulers.DDIM
  alias Margarine.Python.SdxlPythonxServer

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
          pooled_embeds: Nx.Tensor.t() | nil,
          # IMG2IMG parameters
          init_image: String.t() | nil,
          denoising_strength: float()
        }

  @doc """
  Validate generation options.

  Checks all parameters and returns :ok or {:error, reason}.
  """
  @spec validate_opts(keyword()) :: :ok | {:error, String.t()}
  def validate_opts(opts) do
    prompt = Keyword.get(opts, :prompt)
    model = Keyword.get(opts, :model, :sdxl_base)
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
        model = Keyword.get(opts, :model, :sdxl_base)
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
          pooled_embeds: nil,
          # IMG2IMG parameters
          init_image: Keyword.get(opts, :init_image),
          denoising_strength: Keyword.get(opts, :denoising_strength, 0.75)
        }

        {:ok, state}

      error ->
        error
    end
  end

  @doc """
  Execute the complete SDXL generation pipeline.

  Takes a prepared pipeline state and generates an image.

  Handles both text2img and img2img - the only difference is how initial latents are prepared!

  ## Steps:
  1. Start/get SdxlPythonxServer GenServer
  2. Encode text prompt to embeddings (dual CLIP)
  3. Initialize DDIM scheduler with timesteps
  4. Generate initial latents:
     - **text2img**: Random noise
     - **img2img**: Encoded image + noise based on denoising_strength
  5. Denoising loop (UNet forward passes with CFG) - SAME FOR BOTH!
  6. Decode latents to image

  Returns `{:ok, image_tensor}` where image is {H, W, 3} uint8 RGB tensor.
  """
  @spec generate(pipeline_state()) :: {:ok, Nx.Tensor.t()} | {:error, String.t()}
  def generate(state) do
    mode = if state.init_image, do: "img2img", else: "text2img"
    Logger.info("[Margarine.Pipelines.Sdxl] Starting #{mode} generation: #{state.prompt}")
    Logger.info("[Margarine.Pipelines.Sdxl] Model: #{state.model}, Steps: #{state.steps}, Size: #{inspect(state.size)}")

    with {:ok, server} <- get_or_start_server(state.model),
         {:ok, embeds} <- encode_prompt(server, state),
         {:ok, scheduler} <- initialize_scheduler(state),
         {:ok, time_ids} <- get_time_ids(server, state),
         {:ok, latents} <- prepare_latents(server, scheduler, state),
         {:ok, denoised} <- denoising_loop(server, state, scheduler, latents, embeds, time_ids),
         {:ok, image} <- decode_to_image(server, denoised) do
      Logger.info("[Margarine.Pipelines.Sdxl] ✓ Generation complete")
      {:ok, image}
    end
  end

  # Prepare initial latents - the ONLY difference between text2img and img2img!
  defp prepare_latents(server, _scheduler, %{init_image: nil} = state) do
    # text2img: Just random noise
    generate_initial_latents(server, state)
  end

  defp prepare_latents(server, scheduler, %{init_image: init_image, denoising_strength: strength} = state) do
    # Special case: strength=1.0 means complete regeneration from noise
    # Use pure random noise (same as text2img) for consistency
    if strength >= 1.0 do
      Logger.info("[Margarine.Pipelines.Sdxl] Strength=1.0 detected, using pure random noise (text2img mode)")
      generate_initial_latents(server, state)
    else
      # img2img: Encoded image + noise
      case prepare_image_latents(server, scheduler, init_image, strength, state) do
        {:ok, latents, _timestep_idx} -> {:ok, latents}
        error -> error
      end
    end
  end

  # Private pipeline steps

  defp get_or_start_server(model) do
    server_name = server_name_for_model(model)

    case Process.whereis(server_name) do
      nil ->
        Logger.info("[Margarine.Pipelines.Sdxl] Starting SdxlPythonxServer for #{model}...")
        opts = [name: server_name, model: model]

        case SdxlPythonxServer.start_link(opts) do
          {:ok, pid} ->
            Logger.info("[Margarine.Pipelines.Sdxl] ✓ SdxlPythonxServer started: #{inspect(pid)}")
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
    Logger.info("[Margarine.Pipelines.Sdxl] Encoding prompt...")

    case SdxlPythonxServer.encode_prompt(server, state.prompt,
           guidance_scale: state.guidance_scale
         ) do
      {:ok, result} ->
        Logger.info("[Margarine.Pipelines.Sdxl] ✓ Prompt encoded")
        {:ok, result}

      {:error, reason} ->
        {:error, "Prompt encoding failed: #{inspect(reason)}"}
    end
  end

  defp initialize_scheduler(state) do
    Logger.info("[Margarine.Pipelines.Sdxl] Initializing DDIM scheduler...")
    scheduler = DDIM.new(num_inference_steps: state.steps) |> DDIM.set_timesteps()
    Logger.info("[Margarine.Pipelines.Sdxl] ✓ Scheduler initialized with #{state.steps} steps")
    {:ok, scheduler}
  end

  defp get_time_ids(server, state) do
    Logger.info("[Margarine.Pipelines.Sdxl] Getting time IDs...")
    {height, width} = state.size

    case SdxlPythonxServer.get_time_ids(server, height, width, state.guidance_scale) do
      {:ok, time_ids} ->
        Logger.info("[Margarine.Pipelines.Sdxl] ✓ Time IDs generated")
        {:ok, time_ids}

      {:error, reason} ->
        {:error, "Time ID generation failed: #{inspect(reason)}"}
    end
  end

  defp generate_initial_latents(server, state) do
    Logger.info("[Margarine.Pipelines.Sdxl] Generating initial latents...")
    {height, width} = state.size

    case SdxlPythonxServer.generate_latents(server, height, width, state.seed) do
      {:ok, latents} ->
        # Debug: Check if latents are all zeros
        latents_sum = Nx.sum(Nx.abs(latents)) |> Nx.to_number()
        Logger.info("[Margarine.Pipelines.Sdxl] ✓ Initial latents generated (sum of abs values: #{latents_sum})")
        {:ok, latents}

      {:error, reason} ->
        {:error, "Latent generation failed: #{inspect(reason)}"}
    end
  end

  defp prepare_image_latents(server, scheduler, init_image_path, denoising_strength, state) do
    Logger.info("[Margarine.Pipelines.Sdxl] Preparing latents from init image...")
    Logger.info("[Margarine.Pipelines.Sdxl] Init image: #{init_image_path}")
    Logger.info("[Margarine.Pipelines.Sdxl] Denoising strength: #{denoising_strength}")

    {height, width} = state.size
    Logger.info("[Margarine.Pipelines.Sdxl] Target size: #{height}x#{width}")

    with {:ok, image} <- Margarine.Image.load(init_image_path),
         _ = Logger.info("[Margarine.Pipelines.Sdxl] Loaded image shape: #{inspect(Nx.shape(image))}"),
         {:ok, preprocessed} <- Margarine.Image.preprocess_for_vae(image, {height, width}),
         _ = Logger.info("[Margarine.Pipelines.Sdxl] Preprocessed shape: #{inspect(Nx.shape(preprocessed))}"),
         {:ok, clean_latents} <- SdxlPythonxServer.vae_encode(server, preprocessed),
         _ = Logger.info("[Margarine.Pipelines.Sdxl] Encoded latents shape: #{inspect(Nx.shape(clean_latents))}"),
         {:ok, noise} <- SdxlPythonxServer.generate_latents(server, height, width, state.seed),
         _ = Logger.info("[Margarine.Pipelines.Sdxl] Noise shape: #{inspect(Nx.shape(noise))}") do
      # Calculate starting timestep index based on denoising strength
      # strength=1.0 means start at timestep 0 (full noise, equivalent to text2img)
      # strength=0.0 means start at last timestep (no noise, no change)
      # strength=0.7 means start at timestep for 70% noise
      timestep_idx = round((1.0 - denoising_strength) * (state.steps - 1))

      # Add noise to latents using scheduler's add_noise function
      noisy_latents = DDIM.add_noise(scheduler, clean_latents, noise, timestep_idx)

      Logger.info("[Margarine.Pipelines.Sdxl] ✓ Image latents prepared (starting at timestep idx #{timestep_idx})")
      {:ok, noisy_latents, timestep_idx}
    else
      {:error, reason} ->
        {:error, "Image latent preparation failed: #{inspect(reason)}"}
    end
  end

  defp denoising_loop(server, state, scheduler, latents, embeds, time_ids) do
    Logger.info("[Margarine.Pipelines.Sdxl] Starting denoising loop (#{state.steps} steps)...")

    timesteps = scheduler.timesteps |> Nx.to_flat_list()
    num_steps = length(timesteps)

    result =
      Enum.reduce_while(Enum.with_index(timesteps), latents, fn {_timestep, idx}, current_latents ->
        Logger.debug("[Margarine.Pipelines.Sdxl] Step #{idx + 1}/#{num_steps}")

        case SdxlPythonxServer.unet_forward(
               server,
               current_latents,
               timesteps |> Enum.at(idx),
               embeds.prompt_embeds,
               embeds.pooled_embeds,
               time_ids,
               guidance_scale: state.guidance_scale
             ) do
          {:ok, model_output} ->
            # Debug: Check model output
            model_sum = Nx.sum(Nx.abs(model_output)) |> Nx.to_number()
            Logger.debug("[Margarine.Pipelines.Sdxl] Model output sum: #{model_sum}")

            # Apply scheduler step
            next_latents = DDIM.step(scheduler, model_output, idx, current_latents)

            # Debug: Check next latents
            next_sum = Nx.sum(Nx.abs(next_latents)) |> Nx.to_number()
            Logger.debug("[Margarine.Pipelines.Sdxl] Next latents sum: #{next_sum}")

            {:cont, next_latents}

          {:error, reason} ->
            {:halt, {:error, "Denoising step #{idx} failed: #{inspect(reason)}"}}
        end
      end)

    case result do
      {:error, _} = error ->
        error

      final_latents ->
        Logger.info("[Margarine.Pipelines.Sdxl] ✓ Denoising complete")
        {:ok, final_latents}
    end
  end

  defp decode_to_image(server, latents) do
    Logger.info("[Margarine.Pipelines.Sdxl] Decoding latents to image...")

    # Debug: Check latents before decoding
    latents_sum = Nx.sum(Nx.abs(latents)) |> Nx.to_number()
    Logger.debug("[Margarine.Pipelines.Sdxl] Latents sum before VAE decode: #{latents_sum}")

    case SdxlPythonxServer.vae_decode(server, latents) do
      {:ok, image_tensor} ->
        # Debug: Check decoded tensor
        tensor_sum = Nx.sum(Nx.abs(image_tensor)) |> Nx.to_number()
        Logger.debug("[Margarine.Pipelines.Sdxl] Decoded tensor sum: #{tensor_sum}")

        # Convert from [-1, 1] float32 to [0, 255] uint8
        # Shape: [1, 3, H, W] -> [H, W, 3]
        image =
          image_tensor
          |> Nx.add(1.0)
          |> Nx.multiply(127.5)
          |> Nx.clip(0, 255)
          |> Nx.as_type(:u8)
          |> Nx.squeeze(axes: [0])
          |> Nx.transpose(axes: [1, 2, 0])

        # Debug: Check final image
        image_sum = Nx.sum(image) |> Nx.to_number()
        Logger.debug("[Margarine.Pipelines.Sdxl] Final image sum: #{image_sum}")

        Logger.info("[Margarine.Pipelines.Sdxl] ✓ Image decoded")
        {:ok, image}

      {:error, reason} ->
        {:error, "VAE decode failed: #{inspect(reason)}"}
    end
  end

  defp server_name_for_model(model) do
    :"sdxl_pythonx_server_#{model}"
  end

  # Private validation helpers

  defp validate_prompt(nil), do: {:error, "prompt is required"}
  defp validate_prompt(prompt) when is_binary(prompt) and byte_size(prompt) > 0, do: :ok
  defp validate_prompt(_), do: {:error, "prompt must be a non-empty string"}

  defp validate_model(model) when model in [:sdxl_base, :sdxl_turbo], do: :ok
  defp validate_model(model), do: {:error, "model must be :sdxl_base or :sdxl_turbo, got: #{inspect(model)}"}

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
    # Check divisible by 8 (SDXL requirement, same as FLUX)
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
