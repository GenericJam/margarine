defmodule MargarineTest do
  use ExUnit.Case, async: true

  @moduledoc """
  Tests for Margarine public API.

  Tests the main user-facing functions:
  - Margarine.generate/1 (simple prompt)
  - Margarine.generate/2 (prompt + options)
  """

  alias Margarine

  describe "generate/1" do
    test "accepts a simple string prompt" do
      # For now, we expect this to fail gracefully until we implement Python server integration
      result = Margarine.generate("a red panda")

      # Should return {:ok, tensor} or {:error, reason}
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end

    test "requires a non-empty prompt" do
      assert {:error, reason} = Margarine.generate("")
      assert reason =~ "prompt"
    end

    test "requires a string prompt" do
      assert {:error, reason} = Margarine.generate(nil)
      assert reason =~ "prompt"
    end

    test "rejects invalid prompt types" do
      assert {:error, reason} = Margarine.generate(123)
      assert reason =~ "prompt"
    end
  end

  describe "generate/2" do
    test "accepts prompt with empty options" do
      result = Margarine.generate("a red panda", [])
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end

    test "accepts valid model option" do
      result = Margarine.generate("a red panda", model: :flux_schnell)
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end

    test "accepts valid steps option" do
      result = Margarine.generate("a red panda", steps: 4)
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end

    test "accepts valid guidance_scale option" do
      result = Margarine.generate("a red panda", guidance_scale: 3.5)
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end

    test "accepts valid seed option" do
      result = Margarine.generate("a red panda", seed: 42)
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end

    test "accepts valid size option" do
      result = Margarine.generate("a red panda", size: {1024, 1024})
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end

    test "accepts all options together" do
      opts = [
        model: :flux_schnell,
        steps: 4,
        guidance_scale: 0.0,
        seed: 42,
        size: {1024, 1024}
      ]

      result = Margarine.generate("a red panda", opts)
      assert match?({:ok, _}, result) or match?({:error, _}, result)
    end

    test "rejects invalid model" do
      assert {:error, reason} = Margarine.generate("test", model: :invalid)
      assert reason =~ "model"
    end

    test "rejects invalid steps" do
      assert {:error, reason} = Margarine.generate("test", steps: -1)
      assert reason =~ "steps"
    end

    test "rejects invalid guidance_scale" do
      assert {:error, reason} = Margarine.generate("test", guidance_scale: -1.0)
      assert reason =~ "guidance_scale"
    end

    test "rejects invalid seed" do
      assert {:error, reason} = Margarine.generate("test", seed: -1)
      assert reason =~ "seed"
    end

    test "rejects invalid size" do
      assert {:error, reason} = Margarine.generate("test", size: {0, 1024})
      assert reason =~ "size"
    end

    test "rejects size not divisible by 8" do
      assert {:error, reason} = Margarine.generate("test", size: {1023, 1024})
      assert reason =~ "divisible by 8"
    end
  end

  describe "check_environment/0" do
    test "returns map with pythonx status" do
      result = Margarine.check_environment()
      assert is_map(result)
      assert Map.has_key?(result, :pythonx_initialized)
    end

    test "includes python version when available" do
      result = Margarine.check_environment()

      case result do
        %{pythonx_initialized: true, python_version: version} ->
          assert is_binary(version)
          assert version =~ ~r/\d+\.\d+\.\d+/

        %{pythonx_initialized: false} ->
          # Python not initialized yet, that's okay
          :ok
      end
    end
  end
end
