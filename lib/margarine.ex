defmodule Margarine do
  @moduledoc """
  Margarine - AI Image Generation for Elixir

  Margarine brings FLUX image generation to Elixir using Nx, Pythonx, and PyTorch.
  Generate beautiful images from text prompts with a clean, Elixir-native API.

  ## Quick Start

      # Simple generation with defaults
      {:ok, image} = Margarine.generate("a red panda eating bamboo")

      # Advanced generation with options
      {:ok, image} = Margarine.generate("a red panda eating bamboo",
        model: :flux_schnell,
        steps: 4,
        guidance_scale: 0.0,
        seed: 42,
        size: {1024, 1024}
      )

      # Save the image
      Margarine.Image.save(image, "panda.png")

  ## Configuration

  Margarine automatically downloads and installs Python dependencies on first run
  using UV. No manual Python setup required!

  Supported models:
  - `:flux_schnell` - Fast 4-step FLUX model (default)
  - `:flux_dev` - High quality 28-step FLUX model
  - `:sdxl_base` - Stable Diffusion XL base model (20 steps)
  - `:sdxl_turbo` - SDXL Turbo model (1 step, fast)

  ## Options

  - `:model` - Model to use (`:flux_schnell`, `:flux_dev`, `:sdxl_base`, or `:sdxl_turbo`)
  - `:steps` - Number of denoising steps (default: model-specific)
  - `:guidance_scale` - Guidance strength (default: model-specific)
  - `:seed` - Random seed for reproducibility (default: random)
  - `:size` - Image size as `{width, height}` tuple (default: `{1024, 1024}`)

  ## Memory Requirements

  - FLUX Schnell: ~12GB VRAM (GPU) or ~16GB RAM (CPU)
  - FLUX Dev: ~12GB VRAM (GPU) or ~16GB RAM (CPU)
  - SDXL Base: ~7GB VRAM (GPU) or ~10GB RAM (CPU)
  - SDXL Turbo: ~7GB VRAM (GPU) or ~10GB RAM (CPU)

  Use `check_environment/0` to verify your Python environment is ready.
  """

  alias Margarine.Pipeline
  alias Margarine.Pipelines.Sdxl

  @type generation_opts :: [
          model: :flux_schnell | :flux_dev | :sdxl_base | :sdxl_turbo,
          steps: pos_integer(),
          guidance_scale: float(),
          seed: non_neg_integer() | nil,
          size: {pos_integer(), pos_integer()}
        ]

  @doc """
  Generate an image from a text prompt.

  Returns `{:ok, image}` where image is an Nx tensor, or `{:error, reason}`.

  ## Examples

      {:ok, image} = Margarine.generate("a red panda eating bamboo")

      {:ok, image} = Margarine.generate("a serene mountain landscape",
        model: :flux_dev,
        steps: 28,
        seed: 42
      )

  ## Image Format

  The returned image is an Nx tensor with shape `{height, width, 3}` and type
  `{:u, 8}` (RGB values 0-255). Use `Margarine.Image.save/2` to save as PNG/JPEG.
  """
  @spec generate(String.t(), generation_opts()) :: {:ok, Nx.Tensor.t()} | {:error, String.t()}
  def generate(prompt, opts \\ [])

  def generate(prompt, opts) when is_binary(prompt) and is_list(opts) do
    # Merge prompt into opts for validation
    full_opts = Keyword.put(opts, :prompt, prompt)

    # Route to appropriate pipeline based on model
    model = Keyword.get(opts, :model, :flux_schnell)

    case model do
      model when model in [:flux_schnell, :flux_dev] ->
        with {:ok, state} <- Pipeline.prepare(full_opts),
             {:ok, image} <- Pipeline.generate(state) do
          {:ok, image}
        end

      model when model in [:sdxl_base, :sdxl_turbo] ->
        with {:ok, state} <- Sdxl.prepare(full_opts),
             {:ok, image} <- Sdxl.generate(state) do
          {:ok, image}
        end

      _ ->
        {:error, "Unsupported model: #{inspect(model)}. Must be :flux_schnell, :flux_dev, :sdxl_base, or :sdxl_turbo"}
    end
  end

  def generate(prompt, _opts) when not is_binary(prompt) do
    {:error, "prompt must be a non-empty string"}
  end

  @doc """
  Generate an image from a text prompt and an initial image (img2img).

  Takes an existing image and modifies it according to the prompt. The `denoising_strength`
  parameter controls how much the image changes:
  - `1.0` = completely regenerate (equivalent to text2img)
  - `0.7` = moderate changes (70% noise)
  - `0.3` = subtle changes (30% noise)
  - `0.0` = no changes (identity operation)

  ## Parameters

    * `prompt` - Text description of desired changes
    * `init_image` - Path to initial image file (PNG/JPEG)
    * `opts` - Generation options (same as `generate/2` plus:)
      * `:denoising_strength` - How much to change (0.0-1.0, default: 0.75)

  ## Returns

    * `{:ok, image}` - Modified image as Nx tensor
    * `{:error, reason}` - Generation failed

  ## Examples

      # Subtle style change
      {:ok, image} = Margarine.img2img(
        "turn into a watercolor painting",
        "photo.png",
        denoising_strength: 0.3
      )

      # Moderate transformation
      {:ok, image} = Margarine.img2img(
        "convert to anime style",
        "portrait.jpg",
        denoising_strength: 0.7,
        model: :flux_schnell
      )

      # Complete regeneration (equivalent to text2img)
      {:ok, image} = Margarine.img2img(
        "a mountain landscape",
        "noise.png",
        denoising_strength: 1.0
      )

  """
  @spec img2img(String.t(), String.t(), generation_opts()) ::
          {:ok, Nx.Tensor.t()} | {:error, String.t()}
  def img2img(prompt, init_image, opts \\ [])

  def img2img(prompt, init_image, opts)
      when is_binary(prompt) and is_binary(init_image) and is_list(opts) do
    # Validate init_image exists
    unless File.exists?(init_image) do
      {:error, "Init image not found: #{init_image}"}
    else
      # Add init_image and denoising_strength to opts
      denoising_strength = Keyword.get(opts, :denoising_strength, 0.75)

      full_opts =
        opts
        |> Keyword.put(:prompt, prompt)
        |> Keyword.put(:init_image, init_image)
        |> Keyword.put(:denoising_strength, denoising_strength)

      # Route to appropriate pipeline based on model
      model = Keyword.get(opts, :model, :flux_schnell)

      case model do
        model when model in [:flux_schnell, :flux_dev] ->
          # Same pipeline as text2img - just different initial latents!
          with {:ok, state} <- Pipeline.prepare(full_opts),
               {:ok, image} <- Pipeline.generate(state) do
            {:ok, image}
          end

        model when model in [:sdxl_base, :sdxl_turbo] ->
          # Same pipeline as text2img - just different initial latents!
          with {:ok, state} <- Sdxl.prepare(full_opts),
               {:ok, image} <- Sdxl.generate(state) do
            {:ok, image}
          end

        _ ->
          {:error, "Unsupported model: #{inspect(model)}"}
      end
    end
  end

  def img2img(_prompt, _init_image, _opts) do
    {:error, "prompt and init_image must be strings"}
  end

  @doc """
  Check if the Python environment is initialized and ready.

  Returns a map with:
  - `:pythonx_initialized` - Boolean indicating if Python is ready
  - `:python_version` - Python version string (if initialized)

  ## Examples

      iex> Margarine.check_environment()
      %{pythonx_initialized: true, python_version: "3.11.13"}

  This is useful for health checks and debugging. The first call may trigger
  Python environment initialization, which can take 2-5 minutes on first run.
  """
  @spec check_environment() :: map()
  def check_environment do
    Margarine.Application.check_environment()
  end
end
