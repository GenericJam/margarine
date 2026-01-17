defmodule Margarine.Schedulers.DDIMTest do
  use ExUnit.Case, async: true

  alias Margarine.Schedulers.DDIM

  describe "new/1" do
    test "creates scheduler with default options" do
      scheduler = DDIM.new()

      assert scheduler.num_inference_steps == 50
      assert scheduler.num_train_timesteps == 1000
      assert scheduler.beta_schedule == :scaled_linear
      assert scheduler.beta_start == 0.00085
      assert scheduler.beta_end == 0.012
      assert scheduler.clip_sample == false
      assert scheduler.timesteps == nil
      assert scheduler.alphas_cumprod == nil
    end

    test "creates scheduler with custom options" do
      scheduler = DDIM.new(
        num_inference_steps: 20,
        num_train_timesteps: 1000,
        beta_schedule: :linear,
        beta_start: 0.0001,
        beta_end: 0.02,
        clip_sample: true
      )

      assert scheduler.num_inference_steps == 20
      assert scheduler.num_train_timesteps == 1000
      assert scheduler.beta_schedule == :linear
      assert scheduler.beta_start == 0.0001
      assert scheduler.beta_end == 0.02
      assert scheduler.clip_sample == true
    end
  end

  describe "set_timesteps/2" do
    test "sets timesteps and alphas_cumprod" do
      scheduler = DDIM.new(num_inference_steps: 10, num_train_timesteps: 100)
      scheduler = DDIM.set_timesteps(scheduler)

      assert scheduler.timesteps != nil
      assert scheduler.alphas_cumprod != nil

      # Verify timesteps shape
      assert Nx.shape(scheduler.timesteps) == {10}
      assert Nx.type(scheduler.timesteps) == {:s, 64}

      # Verify alphas_cumprod shape
      assert Nx.shape(scheduler.alphas_cumprod) == {100}

      # Timesteps should be in descending order (999, 989, ..., 9)
      timesteps_list = Nx.to_flat_list(scheduler.timesteps)
      assert Enum.at(timesteps_list, 0) > Enum.at(timesteps_list, -1)
    end

    test "overrides num_inference_steps" do
      scheduler = DDIM.new(num_inference_steps: 10)
      scheduler = DDIM.set_timesteps(scheduler, 20)

      assert scheduler.num_inference_steps == 20
      assert Nx.shape(scheduler.timesteps) == {20}
    end

    test "alphas_cumprod values are between 0 and 1" do
      scheduler = DDIM.new(num_inference_steps: 10)
      scheduler = DDIM.set_timesteps(scheduler)

      alphas = Nx.to_flat_list(scheduler.alphas_cumprod)

      assert Enum.all?(alphas, fn alpha -> alpha >= 0.0 and alpha <= 1.0 end)
    end

    test "alphas_cumprod is monotonically decreasing" do
      scheduler = DDIM.new(num_inference_steps: 10)
      scheduler = DDIM.set_timesteps(scheduler)

      alphas = Nx.to_flat_list(scheduler.alphas_cumprod)

      # Check that each value is less than or equal to the previous
      alphas
      |> Enum.chunk_every(2, 1, :discard)
      |> Enum.each(fn [a, b] -> assert a >= b end)
    end
  end

  describe "step/4" do
    setup do
      scheduler = DDIM.new(num_inference_steps: 4, num_train_timesteps: 100)
      scheduler = DDIM.set_timesteps(scheduler)

      # Create sample tensors with some variation (not all broadcast)
      key = Nx.Random.key(123)
      {sample, key} = Nx.Random.normal(key, 0.5, 0.2, shape: {1, 4, 8, 8})
      {model_output, _key} = Nx.Random.normal(key, 0.1, 0.05, shape: {1, 4, 8, 8})

      {:ok, scheduler: scheduler, sample: sample, model_output: model_output}
    end

    test "performs DDIM step", %{scheduler: scheduler, sample: sample, model_output: model_output} do
      next_sample = DDIM.step(scheduler, model_output, 0, sample)

      # Verify output shape matches input
      assert Nx.shape(next_sample) == Nx.shape(sample)
      assert Nx.type(next_sample) == Nx.type(sample)

      # Verify output is different from input (denoising occurred)
      refute Nx.all_close(next_sample, sample, atol: 1.0e-6) |> Nx.to_number() == 1
    end

    test "step reduces noise progressively", %{scheduler: scheduler, sample: sample, model_output: model_output} do
      # Run multiple steps
      step1 = DDIM.step(scheduler, model_output, 0, sample)
      step2 = DDIM.step(scheduler, model_output, 1, step1)
      step3 = DDIM.step(scheduler, model_output, 2, step2)

      # Each step should produce different results
      # Use value comparison instead of all_close for broadcast tensors
      sample_vals = Nx.to_flat_list(sample) |> Enum.take(5)
      step1_vals = Nx.to_flat_list(step1) |> Enum.take(5)
      step2_vals = Nx.to_flat_list(step2) |> Enum.take(5)
      step3_vals = Nx.to_flat_list(step3) |> Enum.take(5)

      refute sample_vals == step1_vals
      refute step1_vals == step2_vals
      refute step2_vals == step3_vals
    end

    test "handles last step correctly", %{scheduler: scheduler, sample: sample, model_output: model_output} do
      # Last step should still work
      last_step_idx = scheduler.num_inference_steps - 1
      next_sample = DDIM.step(scheduler, model_output, last_step_idx, sample)

      assert Nx.shape(next_sample) == Nx.shape(sample)
    end

    test "with clip_sample enabled", _context do
      scheduler = DDIM.new(num_inference_steps: 4, clip_sample: true)
      scheduler = DDIM.set_timesteps(scheduler)

      # Create sample with values outside [-1, 1]
      sample = Nx.broadcast(2.0, {1, 4, 8, 8})
      model_output = Nx.broadcast(0.1, {1, 4, 8, 8})

      next_sample = DDIM.step(scheduler, model_output, 0, sample)

      # Output should be clipped to [-1, 1]
      # (This is an indirect test - we can't guarantee the intermediate pred_x0 was clipped,
      # but we verify the function runs without error)
      assert Nx.shape(next_sample) == Nx.shape(sample)
    end
  end

  describe "add_noise/4" do
    setup do
      scheduler = DDIM.new(num_inference_steps: 10)
      scheduler = DDIM.set_timesteps(scheduler)

      # Use varied tensors instead of broadcast for more realistic testing
      key = Nx.Random.key(456)
      {latents, key} = Nx.Random.normal(key, 0.5, 0.2, shape: {1, 4, 8, 8})
      {noise, _key} = Nx.Random.normal(key, 0.0, 1.0, shape: {1, 4, 8, 8})

      {:ok, scheduler: scheduler, latents: latents, noise: noise}
    end

    test "adds noise at specified timestep", %{scheduler: scheduler, latents: latents, noise: noise} do
      noisy = DDIM.add_noise(scheduler, latents, noise, 0)

      # Verify shape matches
      assert Nx.shape(noisy) == Nx.shape(latents)

      # Verify output is different from both inputs
      # Convert tensor to boolean for proper assertion
      refute Nx.all_close(noisy, latents, atol: 1.0e-6) |> Nx.to_number() == 1
      refute Nx.all_close(noisy, noise, atol: 1.0e-6) |> Nx.to_number() == 1
    end

    test "different timesteps produce different noise levels", %{scheduler: scheduler, latents: latents, noise: noise} do
      # Early timestep (more noise)
      noisy_early = DDIM.add_noise(scheduler, latents, noise, 0)

      # Late timestep (less noise)
      noisy_late = DDIM.add_noise(scheduler, latents, noise, 8)

      # Results should be different (early has more noise, late has less)
      # We use a looser tolerance since broadcasting can create similar patterns
      early_vals = Nx.to_flat_list(noisy_early) |> Enum.take(10)
      late_vals = Nx.to_flat_list(noisy_late) |> Enum.take(10)

      # At least verify they're different values
      refute early_vals == late_vals

      # Late timestep should be closer to original latents (less noise)
      # Calculate distances
      dist_early = Nx.subtract(noisy_early, latents) |> Nx.abs() |> Nx.sum() |> Nx.to_number()
      dist_late = Nx.subtract(noisy_late, latents) |> Nx.abs() |> Nx.sum() |> Nx.to_number()

      assert dist_late < dist_early, "Late timestep should be closer to original (less noise)"
    end

    test "handles edge cases", %{scheduler: scheduler, latents: latents, noise: noise} do
      # First timestep (most noise)
      noisy_first = DDIM.add_noise(scheduler, latents, noise, 0)
      assert Nx.shape(noisy_first) == Nx.shape(latents)

      # Last timestep (least noise)
      last_idx = scheduler.num_inference_steps - 1
      noisy_last = DDIM.add_noise(scheduler, latents, noise, last_idx)
      assert Nx.shape(noisy_last) == Nx.shape(latents)
    end
  end

  describe "deterministic behavior" do
    test "same inputs produce same outputs" do
      scheduler1 = DDIM.new(num_inference_steps: 5)
      scheduler1 = DDIM.set_timesteps(scheduler1)

      scheduler2 = DDIM.new(num_inference_steps: 5)
      scheduler2 = DDIM.set_timesteps(scheduler2)

      sample = Nx.broadcast(0.5, {1, 4, 8, 8})
      model_output = Nx.broadcast(0.1, {1, 4, 8, 8})

      result1 = DDIM.step(scheduler1, model_output, 0, sample)
      result2 = DDIM.step(scheduler2, model_output, 0, sample)

      assert Nx.all_close(result1, result2)
    end
  end

  describe "integration with realistic sizes" do
    test "works with SDXL latent dimensions" do
      # SDXL uses 4 channels, 128x128 latents for 1024x1024 image
      scheduler = DDIM.new(num_inference_steps: 20)
      scheduler = DDIM.set_timesteps(scheduler)

      key = Nx.Random.key(42)
      {latents, key} = Nx.Random.normal(key, shape: {1, 4, 128, 128})
      {model_output, _key} = Nx.Random.normal(key, shape: {1, 4, 128, 128})

      # Should handle realistic sizes without error
      result = DDIM.step(scheduler, model_output, 0, latents)
      assert Nx.shape(result) == {1, 4, 128, 128}
    end

    test "works with batch size > 1" do
      scheduler = DDIM.new(num_inference_steps: 5)
      scheduler = DDIM.set_timesteps(scheduler)

      # Batch of 2
      key = Nx.Random.key(42)
      {latents, key} = Nx.Random.normal(key, shape: {2, 4, 64, 64})
      {model_output, _key} = Nx.Random.normal(key, shape: {2, 4, 64, 64})

      result = DDIM.step(scheduler, model_output, 0, latents)
      assert Nx.shape(result) == {2, 4, 64, 64}
    end
  end
end
