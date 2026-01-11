defmodule Margarine.Config do
  @moduledoc """
  Configuration management for Margarine.

  Provides default values and validation for image generation parameters.
  """

  @defaults %{
    default_model: :flux_schnell,
    default_steps: 4,
    default_guidance_scale: 3.5,
    default_size: {1024, 1024},
    timeout: 60_000,
    enable_telemetry: true
  }

  @valid_models [:flux_schnell, :flux_dev]

  @doc """
  Gets a configuration value.

  Returns the value from application config if set, otherwise returns
  the default value. If no default exists and no fallback is provided,
  returns nil.

  ## Examples

      iex> Margarine.Config.get(:default_model)
      :flux_schnell

      iex> Margarine.Config.get(:nonexistent_key, :fallback)
      :fallback
  """
  def get(key, fallback \\ nil) do
    Application.get_env(:margarine, key, Map.get(@defaults, key, fallback))
  end

  @doc """
  Validates generation options and fills in defaults.

  Returns `{:ok, validated_opts}` if all options are valid,
  or `{:error, reason}` if validation fails.

  ## Validations

  - `:model` - Must be one of #{inspect(@valid_models)}
  - `:steps` - Must be a positive integer
  - `:guidance_scale` - Must be a positive number
  - `:size` - Must be a `{width, height}` tuple with positive integers
  - `:seed` - Must be a positive integer or nil

  ## Examples

      iex> Margarine.Config.validate_generation_opts(model: :flux_schnell, steps: 4)
      {:ok, [model: :flux_schnell, steps: 4, guidance_scale: 3.5, size: {1024, 1024}]}

      iex> Margarine.Config.validate_generation_opts(model: :invalid)
      {:error, "Invalid model: :invalid. Must be one of [:flux_schnell, :flux_dev]"}
  """
  def validate_generation_opts(opts) when is_list(opts) do
    with {:ok, model} <- validate_model(opts[:model]),
         {:ok, steps} <- validate_steps(opts[:steps]),
         {:ok, guidance_scale} <- validate_guidance_scale(opts[:guidance_scale]),
         {:ok, size} <- validate_size(opts[:size]),
         {:ok, seed} <- validate_seed(opts[:seed]) do
      validated =
        opts
        |> Keyword.put(:model, model)
        |> Keyword.put(:steps, steps)
        |> Keyword.put(:guidance_scale, guidance_scale)
        |> Keyword.put(:size, size)
        |> maybe_put(:seed, seed)

      {:ok, validated}
    end
  end

  # Private validation functions

  defp validate_model(nil), do: {:ok, get(:default_model)}
  defp validate_model(model) when model in @valid_models, do: {:ok, model}

  defp validate_model(model) do
    {:error, "Invalid model: #{inspect(model)}. Must be one of #{inspect(@valid_models)}"}
  end

  defp validate_steps(nil), do: {:ok, get(:default_steps)}
  defp validate_steps(steps) when is_integer(steps) and steps > 0, do: {:ok, steps}

  defp validate_steps(steps) do
    {:error, "Invalid steps: #{inspect(steps)}. Must be a positive integer"}
  end

  defp validate_guidance_scale(nil), do: {:ok, get(:default_guidance_scale)}

  defp validate_guidance_scale(scale) when is_number(scale) and scale > 0,
    do: {:ok, scale}

  defp validate_guidance_scale(scale) do
    {:error, "Invalid guidance_scale: #{inspect(scale)}. Must be a positive number"}
  end

  defp validate_size(nil), do: {:ok, get(:default_size)}

  defp validate_size({w, h} = size)
       when is_integer(w) and is_integer(h) and w > 0 and h > 0 do
    {:ok, size}
  end

  defp validate_size(size) do
    {:error,
     "Invalid size: #{inspect(size)}. Must be a {width, height} tuple with positive integers"}
  end

  defp validate_seed(nil), do: {:ok, nil}
  defp validate_seed(seed) when is_integer(seed) and seed >= 0, do: {:ok, seed}

  defp validate_seed(seed) do
    {:error, "Invalid seed: #{inspect(seed)}. Must be a non-negative integer or nil"}
  end

  defp maybe_put(opts, _key, nil), do: opts
  defp maybe_put(opts, key, value), do: Keyword.put(opts, key, value)
end
