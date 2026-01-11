defmodule Margarine.ConfigTest do
  use ExUnit.Case, async: true

  alias Margarine.Config

  describe "get/2" do
    test "returns default values when not configured" do
      # Default model should be :flux_schnell
      assert Config.get(:default_model) == :flux_schnell

      # Default steps should be 4 (FLUX Schnell default)
      assert Config.get(:default_steps) == 4

      # Default guidance scale
      assert Config.get(:default_guidance_scale) == 3.5

      # Default size (1024x1024)
      assert Config.get(:default_size) == {1024, 1024}

      # Default timeout (60 seconds)
      assert Config.get(:timeout) == 60_000

      # Telemetry enabled by default
      assert Config.get(:enable_telemetry) == true
    end

    test "returns custom value with fallback" do
      assert Config.get(:nonexistent_key, :fallback) == :fallback
    end

    test "returns nil for nonexistent keys without fallback" do
      assert Config.get(:nonexistent_key) == nil
    end
  end

  describe "validate_generation_opts/1" do
    test "accepts valid options" do
      opts = [
        model: :flux_schnell,
        steps: 4,
        guidance_scale: 3.5,
        seed: 42,
        size: {1024, 1024}
      ]

      assert {:ok, validated} = Config.validate_generation_opts(opts)
      assert validated[:model] == :flux_schnell
      assert validated[:steps] == 4
      assert validated[:guidance_scale] == 3.5
      assert validated[:seed] == 42
      assert validated[:size] == {1024, 1024}
    end

    test "fills in defaults for missing options" do
      opts = [model: :flux_schnell]

      assert {:ok, validated} = Config.validate_generation_opts(opts)
      assert validated[:steps] == 4
      assert validated[:guidance_scale] == 3.5
      assert validated[:size] == {1024, 1024}
    end

    test "rejects invalid model" do
      opts = [model: :invalid_model]

      assert {:error, reason} = Config.validate_generation_opts(opts)
      assert reason =~ "Invalid model"
    end

    test "rejects invalid steps (must be positive integer)" do
      assert {:error, reason} = Config.validate_generation_opts(steps: 0)
      assert reason =~ "steps"

      assert {:error, reason} = Config.validate_generation_opts(steps: -1)
      assert reason =~ "steps"

      assert {:error, reason} = Config.validate_generation_opts(steps: "4")
      assert reason =~ "steps"
    end

    test "rejects invalid guidance_scale (must be positive number)" do
      assert {:error, reason} = Config.validate_generation_opts(guidance_scale: -1)
      assert reason =~ "guidance_scale"

      assert {:error, reason} = Config.validate_generation_opts(guidance_scale: "3.5")
      assert reason =~ "guidance_scale"
    end

    test "rejects invalid size (must be {width, height} tuple with positive integers)" do
      assert {:error, reason} = Config.validate_generation_opts(size: {-100, 100})
      assert reason =~ "size"

      assert {:error, reason} = Config.validate_generation_opts(size: {100, -100})
      assert reason =~ "size"

      assert {:error, reason} = Config.validate_generation_opts(size: 1024)
      assert reason =~ "size"

      assert {:error, reason} = Config.validate_generation_opts(size: {100, 100, 100})
      assert reason =~ "size"
    end

    test "accepts valid seed (positive integer)" do
      assert {:ok, validated} = Config.validate_generation_opts(seed: 42)
      assert validated[:seed] == 42
    end

    test "accepts nil seed for random generation" do
      assert {:ok, validated} = Config.validate_generation_opts(seed: nil)
      assert validated[:seed] == nil
    end

    test "rejects invalid seed" do
      assert {:error, reason} = Config.validate_generation_opts(seed: -1)
      assert reason =~ "seed"

      assert {:error, reason} = Config.validate_generation_opts(seed: "42")
      assert reason =~ "seed"
    end
  end
end
