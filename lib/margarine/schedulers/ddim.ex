defmodule Margarine.Schedulers.DDIM do
  @moduledoc """
  Denoising Diffusion Implicit Models (DDIM) scheduler.

  Pure Nx implementation - no Python dependencies.

  DDIM is a deterministic scheduler that provides faster sampling than DDPM
  by skipping timesteps. Works with SDXL and other diffusion models.

  Key features:
  - Deterministic (same seed = same output)
  - Supports arbitrary number of steps (not limited to 1000 like DDPM)
  - Alpha/beta noise schedule
  - Classifier-free guidance support

  Based on: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddim.py

  ## Example

      scheduler = DDIM.new(num_inference_steps: 20)
      scheduler = DDIM.set_timesteps(scheduler)

      # In denoising loop:
      next_sample = DDIM.step(scheduler, model_output, timestep_idx, sample)
  """

  import Nx.Defn

  @type t :: %{
          num_inference_steps: pos_integer(),
          num_train_timesteps: pos_integer(),
          beta_schedule: atom(),
          beta_start: float(),
          beta_end: float(),
          clip_sample: boolean(),
          timesteps: Nx.Tensor.t() | nil,
          alphas_cumprod: Nx.Tensor.t() | nil
        }

  @doc """
  Create a new DDIM scheduler.

  ## Options

    * `:num_inference_steps` - Number of denoising steps (default: 50)
    * `:num_train_timesteps` - Number of timesteps the model was trained on (default: 1000)
    * `:beta_schedule` - Noise schedule type (default: :scaled_linear)
    * `:beta_start` - Starting beta value (default: 0.00085)
    * `:beta_end` - Ending beta value (default: 0.012)
    * `:clip_sample` - Whether to clip samples to [-1, 1] (default: false)

  SDXL defaults:
  - num_inference_steps: 20 (base), 1 (turbo)
  - beta_schedule: :scaled_linear
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    num_inference_steps = Keyword.get(opts, :num_inference_steps, 50)
    num_train_timesteps = Keyword.get(opts, :num_train_timesteps, 1000)
    beta_schedule = Keyword.get(opts, :beta_schedule, :scaled_linear)
    beta_start = Keyword.get(opts, :beta_start, 0.00085)
    beta_end = Keyword.get(opts, :beta_end, 0.012)
    clip_sample = Keyword.get(opts, :clip_sample, false)

    %{
      num_inference_steps: num_inference_steps,
      num_train_timesteps: num_train_timesteps,
      beta_schedule: beta_schedule,
      beta_start: beta_start,
      beta_end: beta_end,
      clip_sample: clip_sample,
      timesteps: nil,
      alphas_cumprod: nil
    }
  end

  @doc """
  Set timesteps for inference.

  Calculates the alpha schedule and selects evenly-spaced timesteps
  from the training timesteps.

  ## Parameters

    * `scheduler` - Scheduler state
    * `num_inference_steps` - Optional override for number of steps
  """
  @spec set_timesteps(t(), pos_integer() | nil) :: t()
  def set_timesteps(scheduler, num_inference_steps \\ nil) do
    num_steps = num_inference_steps || scheduler.num_inference_steps

    # Calculate beta schedule
    betas = get_betas(
      scheduler.beta_schedule,
      scheduler.num_train_timesteps,
      scheduler.beta_start,
      scheduler.beta_end
    )

    # Calculate alphas: alpha_t = 1 - beta_t
    alphas = Nx.subtract(1.0, betas)

    # Calculate cumulative product of alphas
    alphas_cumprod = cumulative_product(alphas)

    # Select evenly-spaced timesteps from high to low
    # For num_steps=1: [999]
    # For num_steps=50: [999, 979, ..., 0]
    timesteps =
      if num_steps == 1 do
        # Special case for single step - avoid linspace edge case
        Nx.tensor([scheduler.num_train_timesteps - 1], type: {:s, 64})
      else
        # Use float type for linspace computation, then convert to integer
        Nx.linspace(scheduler.num_train_timesteps - 1, 0, n: num_steps, type: {:f, 32})
        |> Nx.round()
        |> Nx.as_type({:s, 64})
      end

    Map.merge(scheduler, %{
      timesteps: timesteps,
      alphas_cumprod: alphas_cumprod,
      num_inference_steps: num_steps
    })
  end

  @doc """
  Perform one DDIM step.

  DDIM update formula (deterministic):
  ```
  x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1 - alpha_{t-1}) * pred_epsilon
  ```

  Where:
  - `pred_x0` is the predicted original sample (calculated from model output)
  - `pred_epsilon` is the predicted noise

  ## Parameters

    * `scheduler` - Scheduler with timesteps set
    * `model_output` - Predicted noise from model
    * `timestep_idx` - Current timestep index (0 = first step)
    * `sample` - Current latent sample
  """
  @spec step(t(), Nx.Tensor.t(), non_neg_integer(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def step(scheduler, model_output, timestep_idx, sample) do
    if scheduler.clip_sample do
      ddim_step_with_clip(
        sample,
        model_output,
        scheduler.timesteps,
        scheduler.alphas_cumprod,
        timestep_idx,
        scheduler.num_inference_steps
      )
    else
      ddim_step_no_clip(
        sample,
        model_output,
        scheduler.timesteps,
        scheduler.alphas_cumprod,
        timestep_idx,
        scheduler.num_inference_steps
      )
    end
  end

  # Nx.defn for JIT compilation (with clipping)
  defnp ddim_step_with_clip(sample, model_output, timesteps, alphas_cumprod, timestep_idx, num_steps) do
    {pred_original_sample, alpha_prod_t_prev, sqrt_one_minus_alpha_prod_prev} =
      compute_pred_original_sample(sample, model_output, timesteps, alphas_cumprod, timestep_idx, num_steps)

    # Clip predicted x0
    pred_original_sample = Nx.clip(pred_original_sample, -1.0, 1.0)

    # Calculate final prev_sample
    compute_prev_sample(pred_original_sample, model_output, alpha_prod_t_prev, sqrt_one_minus_alpha_prod_prev)
  end

  # Nx.defn for JIT compilation (without clipping)
  defnp ddim_step_no_clip(sample, model_output, timesteps, alphas_cumprod, timestep_idx, num_steps) do
    {pred_original_sample, alpha_prod_t_prev, sqrt_one_minus_alpha_prod_prev} =
      compute_pred_original_sample(sample, model_output, timesteps, alphas_cumprod, timestep_idx, num_steps)

    # Calculate final prev_sample (no clipping)
    compute_prev_sample(pred_original_sample, model_output, alpha_prod_t_prev, sqrt_one_minus_alpha_prod_prev)
  end

  # Shared computation for pred_original_sample
  defnp compute_pred_original_sample(sample, model_output, timesteps, alphas_cumprod, timestep_idx, num_steps) do
    # Get current timestep value
    timestep = timesteps[timestep_idx]

    # Get alpha_prod for current timestep
    alpha_prod_t = alphas_cumprod[timestep]

    # Get alpha_prod for previous timestep (or 1.0 for first step)
    alpha_prod_t_prev =
      if timestep_idx < num_steps - 1 do
        prev_timestep = timesteps[timestep_idx + 1]
        alphas_cumprod[prev_timestep]
      else
        # Last step: alpha_prod_t_prev = 1.0 (clean image)
        Nx.tensor(1.0, type: {:f, 32})
      end

    # Calculate predicted original sample (x_0)
    # pred_x0 = (sample - sqrt(1 - alpha_t) * noise) / sqrt(alpha_t)
    sqrt_alpha_prod = Nx.sqrt(alpha_prod_t)
    sqrt_one_minus_alpha_prod = Nx.sqrt(Nx.subtract(1.0, alpha_prod_t))

    pred_original_sample =
      Nx.subtract(sample, Nx.multiply(sqrt_one_minus_alpha_prod, model_output))
      |> Nx.divide(sqrt_alpha_prod)

    # Calculate direction pointing to x_t for the previous timestep
    sqrt_one_minus_alpha_prod_prev = Nx.sqrt(Nx.subtract(1.0, alpha_prod_t_prev))

    {pred_original_sample, alpha_prod_t_prev, sqrt_one_minus_alpha_prod_prev}
  end

  # Compute final prev_sample
  defnp compute_prev_sample(pred_original_sample, model_output, alpha_prod_t_prev, sqrt_one_minus_alpha_prod_prev) do
    # Calculate direction pointing to x_t
    # dir_xt = sqrt(1 - alpha_{t-1}) * noise
    pred_sample_direction = Nx.multiply(sqrt_one_minus_alpha_prod_prev, model_output)

    # Calculate x_{t-1}
    # x_{t-1} = sqrt(alpha_{t-1}) * pred_x0 + sqrt(1 - alpha_{t-1}) * noise
    sqrt_alpha_prod_prev = Nx.sqrt(alpha_prod_t_prev)

    prev_sample =
      Nx.add(
        Nx.multiply(sqrt_alpha_prod_prev, pred_original_sample),
        pred_sample_direction
      )

    prev_sample
  end

  # Helper: Generate beta schedule
  defp get_betas(:linear, num_train_timesteps, beta_start, beta_end) do
    Nx.linspace(beta_start, beta_end, n: num_train_timesteps, type: {:f, 32})
  end

  defp get_betas(:scaled_linear, num_train_timesteps, beta_start, beta_end) do
    # SDXL uses scaled_linear schedule
    # betas = linspace(sqrt(start), sqrt(end))^2
    Nx.linspace(:math.sqrt(beta_start), :math.sqrt(beta_end), n: num_train_timesteps, type: {:f, 32})
    |> Nx.pow(2)
  end

  # Helper: Cumulative product
  # Nx doesn't have a built-in cumulative product, so we implement it manually
  defp cumulative_product(tensor) do
    flat_list = Nx.to_flat_list(tensor)

    # Build cumulative product iteratively using Elixir
    cumulative_list =
      Enum.scan(flat_list, fn x, acc -> x * acc end)

    # Convert back to tensor
    Nx.tensor(cumulative_list, type: {:f, 32})
  end

  @doc """
  Add noise to latents for img2img.

  This is used to start denoising from a partially noised image.

  ## Parameters

    * `scheduler` - Scheduler with timesteps set
    * `latents` - Original latents (encoded image)
    * `noise` - Random noise to add
    * `timestep_idx` - Which timestep to start from (0 = most noise, num_steps-1 = least noise)

  ## Returns

    * Noised latents at the specified timestep
  """
  @spec add_noise(t(), Nx.Tensor.t(), Nx.Tensor.t(), non_neg_integer()) :: Nx.Tensor.t()
  def add_noise(scheduler, latents, noise, timestep_idx) do
    add_noise_at_timestep(
      latents,
      noise,
      scheduler.timesteps,
      scheduler.alphas_cumprod,
      timestep_idx
    )
  end

  # Nx.defn for adding noise
  defnp add_noise_at_timestep(latents, noise, timesteps, alphas_cumprod, timestep_idx) do
    # Get timestep value
    timestep = timesteps[timestep_idx]

    # Get alpha_prod for this timestep
    alpha_prod_t = alphas_cumprod[timestep]

    # Formula: noisy_latents = sqrt(alpha_t) * latents + sqrt(1 - alpha_t) * noise
    sqrt_alpha_prod = Nx.sqrt(alpha_prod_t)
    sqrt_one_minus_alpha_prod = Nx.sqrt(Nx.subtract(1.0, alpha_prod_t))

    noisy_latents =
      Nx.add(
        Nx.multiply(sqrt_alpha_prod, latents),
        Nx.multiply(sqrt_one_minus_alpha_prod, noise)
      )

    noisy_latents
  end
end
