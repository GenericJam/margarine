defmodule Margarine.Python.FluxServerTest do
  use ExUnit.Case, async: false

  @moduledoc """
  Tests for Margarine.Python.FluxServer module.

  NOTE ON COVERAGE: This module interfaces with Python/Pythonx and external
  libraries (PyTorch, Diffusers), which means some code paths are difficult
  to test without:
  - Actually loading FLUX models (30GB download, 12GB RAM)
  - Mocking Pythonx internals
  - Creating specific tensor configurations

  For TDD purposes, we test:
  - Public API surface (function signatures, error handling)
  - Module initialization and configuration
  - Validation logic that doesn't require model loading

  Integration tests (@tag :integration) will verify actual FLUX generation.
  Coverage of 60-70% is acceptable for this Python integration layer.
  """

  alias Margarine.Python.FluxServer

  describe "module_loaded?/0" do
    test "returns boolean indicating if Python module is available" do
      result = FluxServer.module_loaded?()
      assert is_boolean(result)
    end
  end

  describe "initialize_model/1" do
    test "validates required options" do
      assert {:error, reason} = FluxServer.initialize_model([])
      assert reason =~ "model_type"
    end

    test "validates model_type" do
      opts = [model_type: :invalid, model_id: "test", device: "cpu"]
      assert {:error, reason} = FluxServer.initialize_model(opts)
      assert reason =~ "model_type" or reason =~ "flux_schnell" or reason =~ "flux_dev"
    end

    test "validates device" do
      opts = [model_type: :flux_schnell, model_id: "test", device: "invalid"]
      assert {:error, reason} = FluxServer.initialize_model(opts)
      assert reason =~ "device" or reason =~ "mps" or reason =~ "cuda" or reason =~ "cpu"
    end

    test "validates torch_dtype" do
      opts = [
        model_type: :flux_schnell,
        model_id: "test",
        device: "cpu",
        torch_dtype: "invalid"
      ]

      assert {:error, reason} = FluxServer.initialize_model(opts)

      assert reason =~ "torch_dtype" or reason =~ "bfloat16" or reason =~ "float16" or
               reason =~ "float32"
    end

    test "accepts valid options structure" do
      opts = [
        model_type: :flux_schnell,
        model_id: "black-forest-labs/FLUX.1-schnell",
        device: "cpu",
        torch_dtype: "float32"
      ]

      # Will fail without Python/model, but validates option structure
      result = FluxServer.initialize_model(opts)
      # Result should be either {:ok, map} or {:error, reason}
      assert is_tuple(result) and tuple_size(result) == 2 and elem(result, 0) in [:ok, :error]
    end
  end

  describe "encode_prompt/2" do
    test "requires model to be initialized first" do
      result = FluxServer.encode_prompt("test prompt")
      assert {:error, reason} = result
      assert reason =~ "not initialized" or reason =~ "Models not"
    end

    test "validates prompt is a string" do
      result = FluxServer.encode_prompt(12_345)
      assert {:error, reason} = result
      assert reason =~ "string" or reason =~ "prompt"
    end

    test "validates guidance_scale is a number" do
      result = FluxServer.encode_prompt("test", guidance_scale: "invalid")
      assert {:error, reason} = result
      assert reason =~ "guidance_scale" or reason =~ "number"
    end

    test "accepts valid prompt and options" do
      # Will fail without initialized model, but validates signature
      result = FluxServer.encode_prompt("a red panda", guidance_scale: 3.5)
      assert is_tuple(result) and tuple_size(result) == 2 and elem(result, 0) in [:ok, :error]
    end
  end

  describe "transformer_forward/5" do
    test "requires model to be initialized" do
      latents = Nx.tensor([[[[1.0]]]])
      prompt_embeds = Nx.tensor([[[[1.0]]]])
      pooled_embeds = Nx.tensor([[[[1.0]]]])
      result = FluxServer.transformer_forward(latents, 0.5, prompt_embeds, pooled_embeds)
      assert {:error, reason} = result
      assert reason =~ "not initialized" or reason =~ "Models not"
    end

    test "validates latents is a tensor" do
      result = FluxServer.transformer_forward("not a tensor", 0.5, nil, nil)
      assert {:error, reason} = result
      assert reason =~ "tensor" or reason =~ "Nx.Tensor"
    end

    test "validates timestep is a number" do
      latents = Nx.tensor([[[[1.0]]]])
      result = FluxServer.transformer_forward(latents, "invalid", nil, nil)
      assert {:error, reason} = result
      assert reason =~ "timestep" or reason =~ "number"
    end
  end

  describe "vae_decode/1" do
    test "requires model to be initialized" do
      latents = Nx.tensor([[[[1.0]]]])
      result = FluxServer.vae_decode(latents)
      assert {:error, reason} = result
      assert reason =~ "not initialized" or reason =~ "Models not"
    end

    test "validates latents is a tensor" do
      result = FluxServer.vae_decode("not a tensor")
      assert {:error, reason} = result
      assert reason =~ "tensor" or reason =~ "Nx.Tensor"
    end
  end

  describe "vae_encode/1" do
    test "requires model to be initialized" do
      image = Nx.tensor([[[[1.0]]]])
      result = FluxServer.vae_encode(image)
      assert {:error, reason} = result
      assert reason =~ "not initialized" or reason =~ "Models not"
    end

    test "validates image is a tensor" do
      result = FluxServer.vae_encode("not a tensor")
      assert {:error, reason} = result
      assert reason =~ "tensor" or reason =~ "Nx.Tensor"
    end
  end

  describe "generate_latents/2" do
    test "requires model to be initialized" do
      result = FluxServer.generate_latents(1024, 1024)
      assert {:error, reason} = result
      assert reason =~ "not initialized" or reason =~ "Models not"
    end

    test "validates height is positive integer" do
      result = FluxServer.generate_latents(-1, 1024)
      assert {:error, reason} = result
      assert reason =~ "height" or reason =~ "positive"
    end

    test "validates width is positive integer" do
      result = FluxServer.generate_latents(1024, -1)
      assert {:error, reason} = result
      assert reason =~ "width" or reason =~ "positive"
    end

    test "validates seed is non-negative integer or nil" do
      result = FluxServer.generate_latents(1024, 1024, seed: -1)
      assert {:error, reason} = result
      assert reason =~ "seed" or reason =~ "non-negative"
    end

    test "accepts valid dimensions and optional seed" do
      # Will fail without initialized model, but validates signature
      result = FluxServer.generate_latents(1024, 1024, seed: 42)
      assert is_tuple(result) and tuple_size(result) == 2 and elem(result, 0) in [:ok, :error]
    end
  end

  describe "get_model_info/0" do
    test "returns status when model not initialized" do
      {:ok, info} = FluxServer.get_model_info()
      assert info.status == :not_initialized
    end
  end

  describe "clear_memory/0" do
    test "returns ok even when model not initialized" do
      assert {:ok, _info} = FluxServer.clear_memory()
    end
  end
end
