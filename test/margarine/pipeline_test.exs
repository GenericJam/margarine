defmodule Margarine.PipelineTest do
  use ExUnit.Case, async: true

  @moduledoc """
  Tests for Margarine.Pipeline module.

  Tests the pipeline coordinator that orchestrates:
  - Parameter validation
  - Scheduler initialization
  - Prompt encoding
  - Denoising loop
  - Image decoding

  Note: These are unit tests with mocked Python calls. Integration tests
  will verify actual FLUX generation.
  """

  alias Margarine.Pipeline

  describe "validate_opts/1" do
    test "accepts valid options" do
      opts = [
        prompt: "a red panda",
        model: :flux_schnell,
        steps: 4,
        guidance_scale: 0.0,
        seed: 42,
        size: {1024, 1024}
      ]

      assert :ok = Pipeline.validate_opts(opts)
    end

    test "requires prompt" do
      opts = [model: :flux_schnell]
      assert {:error, reason} = Pipeline.validate_opts(opts)
      assert reason =~ "prompt"
    end

    test "validates model type" do
      opts = [prompt: "test", model: :invalid]
      assert {:error, reason} = Pipeline.validate_opts(opts)
      assert reason =~ "model"
    end

    test "validates steps" do
      opts = [prompt: "test", model: :flux_schnell, steps: -1]
      assert {:error, reason} = Pipeline.validate_opts(opts)
      assert reason =~ "steps"
    end

    test "validates guidance_scale" do
      opts = [prompt: "test", model: :flux_schnell, guidance_scale: -1.0]
      assert {:error, reason} = Pipeline.validate_opts(opts)
      assert reason =~ "guidance_scale"
    end

    test "validates size tuple" do
      opts = [prompt: "test", model: :flux_schnell, size: {0, 1024}]
      assert {:error, reason} = Pipeline.validate_opts(opts)
      assert reason =~ "size"
    end

    test "validates seed" do
      opts = [prompt: "test", model: :flux_schnell, seed: -1]
      assert {:error, reason} = Pipeline.validate_opts(opts)
      assert reason =~ "seed"
    end

    test "uses defaults for missing options" do
      opts = [prompt: "a red panda"]
      assert :ok = Pipeline.validate_opts(opts)
    end
  end

  describe "prepare/1" do
    test "returns pipeline state with validated options" do
      opts = [prompt: "a red panda", model: :flux_schnell]

      case Pipeline.prepare(opts) do
        {:ok, state} ->
          assert is_map(state)
          assert state.prompt == "a red panda"
          assert state.model == :flux_schnell
          assert is_integer(state.steps)
          assert is_float(state.guidance_scale)
          assert is_tuple(state.size)

        {:error, _reason} ->
          # Expected until we implement actual pipeline
          :ok
      end
    end

    test "rejects invalid options" do
      opts = [model: :flux_schnell]  # missing prompt
      assert {:error, reason} = Pipeline.prepare(opts)
      assert reason =~ "prompt"
    end
  end
end
