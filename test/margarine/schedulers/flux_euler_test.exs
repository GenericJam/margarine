defmodule Margarine.Schedulers.FluxEulerTest do
  use ExUnit.Case, async: true

  @moduledoc """
  Tests for FLUX Euler scheduler.

  Pure Nx implementation - no Python dependencies.
  """

  alias Margarine.Schedulers.FluxEuler

  describe "new/1" do
    test "creates scheduler with default options" do
      scheduler = FluxEuler.new()
      assert is_map(scheduler)
      assert scheduler.num_inference_steps == 28
      assert scheduler.shift == 1.0
    end

    test "creates scheduler with custom options" do
      scheduler = FluxEuler.new(num_inference_steps: 4, shift: 2.0)
      assert scheduler.num_inference_steps == 4
      assert scheduler.shift == 2.0
    end
  end

  describe "set_timesteps/2" do
    test "sets timesteps from 1.0 to 0.0" do
      scheduler = FluxEuler.new(num_inference_steps: 4)
      scheduler = FluxEuler.set_timesteps(scheduler)

      assert %Nx.Tensor{} = scheduler.timesteps
      assert Nx.shape(scheduler.timesteps) == {4}

      timesteps_list = Nx.to_list(scheduler.timesteps)
      assert List.first(timesteps_list) == 1.0
      assert List.last(timesteps_list) == 0.0
    end

    test "computes sigmas" do
      scheduler = FluxEuler.new(num_inference_steps: 4)
      scheduler = FluxEuler.set_timesteps(scheduler)

      assert %Nx.Tensor{} = scheduler.sigmas
      assert Nx.shape(scheduler.sigmas) == {4}
    end

    test "allows custom num_inference_steps" do
      scheduler = FluxEuler.new()
      scheduler = FluxEuler.set_timesteps(scheduler, 10)

      assert scheduler.num_inference_steps == 10
      assert Nx.shape(scheduler.timesteps) == {10}
    end
  end

  describe "step/4" do
    test "performs Euler step" do
      scheduler = FluxEuler.new(num_inference_steps: 4)
      scheduler = FluxEuler.set_timesteps(scheduler)

      # Create dummy tensors
      model_output = Nx.tensor([[[[0.1]]]])
      sample = Nx.tensor([[[[1.0]]]])

      # Perform step
      next_sample = FluxEuler.step(scheduler, model_output, 0, sample)

      assert %Nx.Tensor{} = next_sample
      assert Nx.shape(next_sample) == Nx.shape(sample)
    end

    test "step output is different from input" do
      scheduler = FluxEuler.new(num_inference_steps: 4)
      scheduler = FluxEuler.set_timesteps(scheduler)

      model_output = Nx.tensor([[[[0.5]]]])
      sample = Nx.tensor([[[[1.0]]]])

      next_sample = FluxEuler.step(scheduler, model_output, 0, sample)

      # Should be different due to Euler update
      refute Nx.equal(next_sample, sample) |> Nx.all() |> Nx.to_number() == 1
    end
  end

  describe "add_noise/3" do
    test "implements rectified flow noising formula" do
      # Formula: x_t = (1 - t) * latents + t * noise
      # Test with known values to verify correctness

      latents = Nx.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]])
      noise = Nx.tensor([[[[0.0, 0.0]], [[0.0, 0.0]]]])
      timestep = 0.5

      result = FluxEuler.add_noise(latents, noise, timestep)

      # At t=0.5: result should be 0.5 * latents + 0.5 * noise = 0.5 * latents
      expected = Nx.multiply(latents, 0.5)

      assert_in_delta Nx.to_number(result[0][0][0][0]), Nx.to_number(expected[0][0][0][0]), 0.001
      assert_in_delta Nx.to_number(result[0][0][0][1]), Nx.to_number(expected[0][0][0][1]), 0.001
    end

    test "at timestep 0.0 returns pure latents (no noise)" do
      # t=0.0: x_t = (1 - 0) * latents + 0 * noise = latents
      latents = Nx.tensor([[[[5.0, 10.0]], [[15.0, 20.0]]]])
      noise = Nx.tensor([[[[100.0, 200.0]], [[300.0, 400.0]]]])
      timestep = 0.0

      result = FluxEuler.add_noise(latents, noise, timestep)

      # Result should equal latents
      assert Nx.all_close(result, latents) |> Nx.to_number() == 1
    end

    test "at timestep 1.0 returns pure noise (full noise)" do
      # t=1.0: x_t = (1 - 1) * latents + 1 * noise = noise
      latents = Nx.tensor([[[[5.0, 10.0]], [[15.0, 20.0]]]])
      noise = Nx.tensor([[[[100.0, 200.0]], [[300.0, 400.0]]]])
      timestep = 1.0

      result = FluxEuler.add_noise(latents, noise, timestep)

      # Result should equal noise
      assert Nx.all_close(result, noise) |> Nx.to_number() == 1
    end

    test "at timestep 0.3 is 70% latents + 30% noise" do
      # For img2img with strength=0.3, we start at t=0.3
      # x_t = 0.7 * latents + 0.3 * noise
      latents = Nx.tensor([[[[10.0]]]])
      noise = Nx.tensor([[[[0.0]]]])
      timestep = 0.3

      result = FluxEuler.add_noise(latents, noise, timestep)

      # Result should be 7.0 (0.7 * 10.0)
      assert_in_delta Nx.to_number(result[0][0][0][0]), 7.0, 0.01
    end

    test "at timestep 0.7 is 30% latents + 70% noise" do
      # For img2img with strength=0.7, we start at t=0.7
      # x_t = 0.3 * latents + 0.7 * noise
      latents = Nx.tensor([[[[100.0]]]])
      noise = Nx.tensor([[[[10.0]]]])
      timestep = 0.7

      result = FluxEuler.add_noise(latents, noise, timestep)

      # Result should be 37.0 (0.3 * 100.0 + 0.7 * 10.0)
      expected = 0.3 * 100.0 + 0.7 * 10.0
      assert_in_delta Nx.to_number(result[0][0][0][0]), expected, 0.01
    end

    test "preserves tensor shape" do
      # Test with realistic latent shape: {1, 16, 64, 64}
      latents = Nx.broadcast(1.0, {1, 16, 64, 64}) |> Nx.as_type(:f32)
      noise = Nx.broadcast(0.5, {1, 16, 64, 64}) |> Nx.as_type(:f32)
      timestep = 0.5

      result = FluxEuler.add_noise(latents, noise, timestep)

      assert Nx.shape(result) == {1, 16, 64, 64}
      assert Nx.type(result) == {:f, 32}
    end

    test "works with negative latent values" do
      # VAE latents can be negative
      latents = Nx.tensor([[[[-5.0, -3.0]], [[2.0, 4.0]]]])
      noise = Nx.tensor([[[[1.0, 2.0]], [[-1.0, -2.0]]]])
      timestep = 0.5

      result = FluxEuler.add_noise(latents, noise, timestep)

      # Should handle negative values correctly
      # x_t = 0.5 * latents + 0.5 * noise
      expected_0_0 = 0.5 * (-5.0) + 0.5 * 1.0  # -2.0
      expected_0_1 = 0.5 * (-3.0) + 0.5 * 2.0  # -0.5

      assert_in_delta Nx.to_number(result[0][0][0][0]), expected_0_0, 0.01
      assert_in_delta Nx.to_number(result[0][0][0][1]), expected_0_1, 0.01
    end

    test "linear interpolation is correct at multiple timesteps" do
      # Verify linear interpolation property at various points
      latents = Nx.tensor([[[[100.0]]]])
      noise = Nx.tensor([[[[0.0]]]])

      # Test several timesteps
      for t <- [0.0, 0.25, 0.5, 0.75, 1.0] do
        result = FluxEuler.add_noise(latents, noise, t)
        expected = (1.0 - t) * 100.0  # 0 * t term is 0
        assert_in_delta Nx.to_number(result[0][0][0][0]), expected, 0.01
      end
    end

    test "works with different shaped tensors" do
      # Test small latent shape
      latents = Nx.broadcast(2.0, {1, 4, 8, 8}) |> Nx.as_type(:f32)
      noise = Nx.broadcast(1.0, {1, 4, 8, 8}) |> Nx.as_type(:f32)

      result = FluxEuler.add_noise(latents, noise, 0.4)

      assert Nx.shape(result) == {1, 4, 8, 8}
      # At t=0.4: x_t = 0.6 * 2.0 + 0.4 * 1.0 = 1.6
      assert_in_delta Nx.to_number(result[0][0][0][0]), 1.6, 0.01
    end
  end
end
