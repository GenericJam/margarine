defmodule Margarine.Schedulers.FluxEuler do
  @moduledoc """
  Flow Matching Euler Discrete Scheduler for FLUX models.

  Pure Nx implementation - no Python dependencies.

  FLUX uses rectified flow instead of traditional diffusion:
  - Timesteps go from 1.0 (pure noise) to 0.0 (clean image)
  - Simpler math than DDPM/DDIM (Euler method for ODEs)
  - No alpha/beta schedules needed

  Based on: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_flow_match_euler_discrete.py

  ## Example

      scheduler = FluxEuler.new(num_inference_steps: 4)
      scheduler = FluxEuler.set_timesteps(scheduler)

      # In denoising loop:
      next_sample = FluxEuler.step(scheduler, model_output, step_idx, sample)
  """

  import Nx.Defn

  @type t :: %{
          num_inference_steps: pos_integer(),
          shift: float(),
          timesteps: Nx.Tensor.t() | nil,
          sigmas: Nx.Tensor.t() | nil
        }

  @doc """
  Create a new FLUX Euler scheduler.

  ## Options

    * `:num_inference_steps` - Number of denoising steps (default: 28)
    * `:shift` - Timestep shift parameter (default: 1.0)

  FLUX defaults:
  - flux_dev: 28 steps
  - flux_schnell: 4 steps
  """
  @spec new(keyword()) :: t()
  def new(opts \\ []) do
    num_inference_steps = Keyword.get(opts, :num_inference_steps, 28)
    shift = Keyword.get(opts, :shift, 1.0)

    %{
      num_inference_steps: num_inference_steps,
      shift: shift,
      timesteps: nil,
      sigmas: nil
    }
  end

  @doc """
  Set timesteps for inference.

  FLUX uses flow matching, so timesteps go from 1.0 to 0.0:
  - 1.0 = pure noise
  - 0.5 = half noise, half signal
  - 0.0 = clean image

  ## Parameters

    * `scheduler` - Scheduler state
    * `num_inference_steps` - Optional override for number of steps
  """
  @spec set_timesteps(t(), pos_integer() | nil) :: t()
  def set_timesteps(scheduler, num_inference_steps \\ nil) do
    num_steps = num_inference_steps || scheduler.num_inference_steps
    shift = scheduler.shift

    # Create evenly spaced timesteps from 1.0 to 0.0
    timesteps = Nx.linspace(1.0, 0.0, n: num_steps, type: {:f, 32})

    # Apply shift if needed (for dynamic shift based on resolution)
    timesteps =
      if shift != 1.0 do
        apply_shift(timesteps, shift)
      else
        timesteps
      end

    # Compute sigmas from timesteps
    # Following diffusers: sigmas are just the normalized timesteps
    # This avoids infinity issues with sigma = t / (1-t) formula
    sigmas = timesteps

    Map.merge(scheduler, %{
      timesteps: timesteps,
      sigmas: sigmas,
      num_inference_steps: num_steps
    })
  end

  @doc """
  Perform one Euler step.

  Flow matching uses a simpler update rule than DDPM/DDIM:
  `x_{t-1} = x_t - dt * model_output`

  Where:
  - `x_t` is current sample
  - `model_output` is the predicted velocity/noise
  - `dt` is the step size

  ## Parameters

    * `scheduler` - Scheduler with timesteps set
    * `model_output` - Predicted noise/velocity from model
    * `timestep_idx` - Current timestep index
    * `sample` - Current latent sample
  """
  @spec step(t(), Nx.Tensor.t(), non_neg_integer(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def step(scheduler, model_output, timestep_idx, sample) do
    euler_step(
      sample,
      model_output,
      scheduler.sigmas,
      timestep_idx,
      scheduler.num_inference_steps
    )
  end

  # Nx.defn for JIT compilation
  defnp euler_step(sample, model_output, sigmas, timestep_idx, num_steps) do
    # Get current sigma
    sigma = sigmas[timestep_idx]

    # Calculate dt (step size)
    dt =
      if timestep_idx < num_steps - 1 do
        # dt = sigma_{t} - sigma_{t+1}
        next_sigma = sigmas[timestep_idx + 1]
        sigma - next_sigma
      else
        # Last step: dt = sigma (go all the way to 0)
        sigma
      end

    # Euler step: x_{t-1} = x_t - dt * model_output
    Nx.subtract(sample, Nx.multiply(dt, model_output))
  end

  @doc """
  Add noise to latents for img2img.

  For FLUX's rectified flow, noising is linear interpolation:
  `x_t = (1 - t) * x_0 + t * noise`

  Where:
  - `x_0` is the clean latent (encoded from init image)
  - `noise` is random Gaussian noise
  - `t` is the timestep (0.0 = clean, 1.0 = pure noise)
  - `x_t` is the noisy latent

  ## Parameters

    * `latents` - Clean latents from VAE encoder
    * `noise` - Random noise (same shape as latents)
    * `timestep` - Noise level (0.0 to 1.0)

  ## Returns

    Noisy latents ready for denoising

  ## Examples

      # For img2img with strength=0.7, start at timestep 0.7
      noisy_latents = FluxEuler.add_noise(clean_latents, noise, 0.7)
      # Result is 30% clean image, 70% noise

  """
  @spec add_noise(Nx.Tensor.t(), Nx.Tensor.t(), float()) :: Nx.Tensor.t()
  def add_noise(latents, noise, timestep) when is_float(timestep) do
    add_noise_impl(latents, noise, timestep)
  end

  # Nx.defn for JIT compilation
  defnp add_noise_impl(latents, noise, timestep) do
    # Rectified flow: x_t = (1 - t) * x_0 + t * noise
    # This is linear interpolation between clean latents and noise
    t = Nx.as_type(timestep, {:f, 32})
    one_minus_t = Nx.subtract(1.0, t)

    # (1 - t) * latents
    scaled_latents = Nx.multiply(one_minus_t, latents)
    # t * noise
    scaled_noise = Nx.multiply(t, noise)

    # x_t = (1 - t) * latents + t * noise
    Nx.add(scaled_latents, scaled_noise)
  end

  # Helper for applying timestep shift
  defnp apply_shift(timesteps, shift) do
    # Shift formula: timestep_shifted = shift * timestep / (1 + (shift - 1) * timestep)
    # Convert shift to tensor outside defn or use as scalar
    shift_tensor = Nx.as_type(shift, {:f, 32})

    numerator = Nx.multiply(shift_tensor, timesteps)

    denominator =
      Nx.add(
        1.0,
        Nx.multiply(Nx.subtract(shift_tensor, 1.0), timesteps)
      )

    Nx.divide(numerator, denominator)
  end
end
