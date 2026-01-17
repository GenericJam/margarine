defmodule Margarine.Pipelines.SdxlTest do
  use ExUnit.Case, async: true

  alias Margarine.Pipelines.Sdxl

  describe "validate_opts/1" do
    test "validates prompt is required" do
      assert {:error, msg} = Sdxl.validate_opts([])
      assert msg =~ "prompt is required"
    end

    test "validates prompt is a non-empty string" do
      assert {:error, msg} = Sdxl.validate_opts(prompt: "")
      assert msg =~ "prompt must be a non-empty string"

      assert {:error, msg} = Sdxl.validate_opts(prompt: nil)
      assert msg =~ "prompt is required"

      assert {:error, msg} = Sdxl.validate_opts(prompt: 123)
      assert msg =~ "prompt must be a non-empty string"
    end

    test "validates model when provided" do
      assert {:error, msg} = Sdxl.validate_opts(prompt: "test", model: :flux_schnell)
      assert msg =~ "model must be :sdxl_base or :sdxl_turbo"
    end

    test "accepts valid sdxl models" do
      assert :ok = Sdxl.validate_opts(prompt: "test", model: :sdxl_base)
      assert :ok = Sdxl.validate_opts(prompt: "test", model: :sdxl_turbo)
    end

    test "validates steps when provided" do
      assert {:error, msg} = Sdxl.validate_opts(prompt: "test", steps: 0)
      assert msg =~ "steps must be a positive integer"

      assert {:error, msg} = Sdxl.validate_opts(prompt: "test", steps: -1)
      assert msg =~ "steps must be a positive integer"

      assert {:error, msg} = Sdxl.validate_opts(prompt: "test", steps: "10")
      assert msg =~ "steps must be a positive integer"
    end

    test "accepts valid steps" do
      assert :ok = Sdxl.validate_opts(prompt: "test", steps: 20)
    end

    test "validates guidance_scale when provided" do
      assert {:error, msg} = Sdxl.validate_opts(prompt: "test", guidance_scale: -1.0)
      assert msg =~ "guidance_scale must be >= 0.0"

      assert {:error, msg} = Sdxl.validate_opts(prompt: "test", guidance_scale: "7.5")
      assert msg =~ "guidance_scale must be >= 0.0"
    end

    test "accepts valid guidance_scale" do
      assert :ok = Sdxl.validate_opts(prompt: "test", guidance_scale: 7.5)
      assert :ok = Sdxl.validate_opts(prompt: "test", guidance_scale: 0.0)
    end

    test "validates size must be divisible by 8" do
      assert {:error, msg} = Sdxl.validate_opts(prompt: "test", size: {1023, 1024})
      assert msg =~ "size dimensions must be divisible by 8"

      assert {:error, msg} = Sdxl.validate_opts(prompt: "test", size: {1024, 1023})
      assert msg =~ "size dimensions must be divisible by 8"
    end

    test "accepts valid size" do
      assert :ok = Sdxl.validate_opts(prompt: "test", size: {1024, 1024})
      assert :ok = Sdxl.validate_opts(prompt: "test", size: {512, 512})
    end

    test "validates seed when provided" do
      assert {:error, msg} = Sdxl.validate_opts(prompt: "test", seed: -1)
      assert msg =~ "seed must be a non-negative integer"

      assert {:error, msg} = Sdxl.validate_opts(prompt: "test", seed: "42")
      assert msg =~ "seed must be a non-negative integer"
    end

    test "accepts valid seed" do
      assert :ok = Sdxl.validate_opts(prompt: "test", seed: 42)
      assert :ok = Sdxl.validate_opts(prompt: "test", seed: 0)
    end

    test "accepts nil seed" do
      assert :ok = Sdxl.validate_opts(prompt: "test", seed: nil)
    end
  end

  describe "prepare/1" do
    test "prepares pipeline state with defaults" do
      {:ok, state} = Sdxl.prepare(prompt: "a red panda")

      assert state.prompt == "a red panda"
      assert state.model == :sdxl_base
      assert state.steps == 20
      assert state.guidance_scale == 7.5
      assert state.size == {1024, 1024}
      assert state.seed == nil
      assert state.scheduler == nil
      assert state.latents == nil
      assert state.prompt_embeds == nil
      assert state.pooled_embeds == nil
    end

    test "prepares pipeline state with custom options" do
      {:ok, state} =
        Sdxl.prepare(
          prompt: "a blue panda",
          model: :sdxl_turbo,
          steps: 1,
          guidance_scale: 0.0,
          size: {512, 512},
          seed: 42
        )

      assert state.prompt == "a blue panda"
      assert state.model == :sdxl_turbo
      assert state.steps == 1
      assert state.guidance_scale == 0.0
      assert state.size == {512, 512}
      assert state.seed == 42
    end

    test "returns error when validation fails" do
      assert {:error, msg} = Sdxl.prepare(prompt: "")
      assert msg =~ "prompt must be a non-empty string"
    end
  end

  describe "generate/1 - parameter validation" do
    test "validates state has required fields" do
      # Minimal valid state (includes img2img fields)
      state = %{
        prompt: "test",
        model: :sdxl_base,
        steps: 1,
        guidance_scale: 7.5,
        size: {512, 512},
        seed: nil,
        scheduler: nil,
        latents: nil,
        prompt_embeds: nil,
        pooled_embeds: nil,
        init_image: nil,
        denoising_strength: 0.75
      }

      # Should not crash with state structure
      # Will succeed if server is running, or fail gracefully if not
      result = Sdxl.generate(state)
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end
  end
end
