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
end
