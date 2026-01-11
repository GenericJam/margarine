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
