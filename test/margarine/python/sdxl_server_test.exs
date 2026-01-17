defmodule Margarine.Python.SdxlServerTest do
  use ExUnit.Case, async: true

  alias Margarine.Python.SdxlServer

  describe "module_loaded?/0" do
    test "returns true when Python module file exists" do
      assert SdxlServer.module_loaded?() == true
    end
  end

  describe "parameter validation" do
    test "initialize_model validates model_type" do
      assert {:error, msg} = SdxlServer.initialize_model(
        model_type: nil,
        model_id: "stabilityai/stable-diffusion-xl-base-1.0"
      )
      assert msg =~ "model_type is required"
    end

    test "initialize_model validates model_id" do
      assert {:error, msg} = SdxlServer.initialize_model(
        model_type: :sdxl_base,
        model_id: nil
      )
      assert msg =~ "model_id is required"
    end

    test "initialize_model validates model_type value" do
      assert {:error, msg} = SdxlServer.initialize_model(
        model_type: :invalid_model,
        model_id: "stabilityai/stable-diffusion-xl-base-1.0"
      )
      assert msg =~ "model_type must be :sdxl_base or :sdxl_turbo"
    end

    test "initialize_model validates device" do
      assert {:error, msg} = SdxlServer.initialize_model(
        model_type: :sdxl_base,
        model_id: "stabilityai/stable-diffusion-xl-base-1.0",
        device: "invalid"
      )
      assert msg =~ "device must be mps, cuda, or cpu"
    end

    test "initialize_model validates torch_dtype" do
      assert {:error, msg} = SdxlServer.initialize_model(
        model_type: :sdxl_base,
        model_id: "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype: "invalid"
      )
      assert msg =~ "torch_dtype must be float16, float32, or bfloat16"
    end
  end

  describe "encode_prompt/2" do
    test "validates prompt is a string" do
      assert {:error, msg} = SdxlServer.encode_prompt(nil)
      assert msg == "Prompt must be a string"

      assert {:error, msg} = SdxlServer.encode_prompt(123)
      assert msg == "Prompt must be a string"
    end

    test "validates prompt is not empty" do
      assert {:error, msg} = SdxlServer.encode_prompt("")
      assert msg =~ "Prompt cannot be empty"
    end

    test "validates guidance_scale is a number" do
      assert {:error, msg} = SdxlServer.encode_prompt("test prompt", guidance_scale: "not_a_number")
      assert msg =~ "guidance_scale must be a number"
    end

    test "accepts valid prompt with default options" do
      # Should return error about models not initialized (not validation error)
      assert {:error, msg} = SdxlServer.encode_prompt("a red panda")
      assert msg =~ "Models not initialized"
    end

    test "accepts valid prompt with custom guidance_scale" do
      assert {:error, msg} = SdxlServer.encode_prompt("a red panda", guidance_scale: 7.5)
      assert msg =~ "Models not initialized"
    end
  end

  describe "unet_forward/5" do
    setup do
      # Create mock tensors
      latents = Nx.broadcast(0.5, {1, 4, 8, 8})
      prompt_embeds = Nx.broadcast(0.1, {1, 77, 2048})
      pooled_embeds = Nx.broadcast(0.1, {1, 1280})

      {:ok, latents: latents, prompt_embeds: prompt_embeds, pooled_embeds: pooled_embeds}
    end

    test "validates latents is a tensor", %{prompt_embeds: prompt_embeds, pooled_embeds: pooled_embeds} do
      assert {:error, msg} = SdxlServer.unet_forward("not_a_tensor", 500, prompt_embeds, pooled_embeds)
      assert msg == "latents must be an Nx.Tensor"
    end

    test "validates timestep is a number", %{latents: latents, prompt_embeds: prompt_embeds, pooled_embeds: pooled_embeds} do
      assert {:error, msg} = SdxlServer.unet_forward(latents, "not_a_number", prompt_embeds, pooled_embeds)
      assert msg == "timestep must be a number"
    end

    test "validates prompt_embeds is a tensor", %{latents: latents, pooled_embeds: pooled_embeds} do
      assert {:error, msg} = SdxlServer.unet_forward(latents, 500, "not_a_tensor", pooled_embeds)
      assert msg == "prompt_embeds must be an Nx.Tensor"
    end

    test "validates pooled_embeds is a tensor", %{latents: latents, prompt_embeds: prompt_embeds} do
      assert {:error, msg} = SdxlServer.unet_forward(latents, 500, prompt_embeds, "not_a_tensor")
      assert msg == "pooled_embeds must be an Nx.Tensor"
    end

    test "accepts valid inputs", %{latents: latents, prompt_embeds: prompt_embeds, pooled_embeds: pooled_embeds} do
      # Should return error about models not initialized (not validation error)
      assert {:error, msg} = SdxlServer.unet_forward(latents, 500, prompt_embeds, pooled_embeds)
      assert msg =~ "Models not initialized"
    end

    test "accepts custom guidance_scale", %{latents: latents, prompt_embeds: prompt_embeds, pooled_embeds: pooled_embeds} do
      assert {:error, msg} = SdxlServer.unet_forward(
        latents,
        500,
        prompt_embeds,
        pooled_embeds,
        guidance_scale: 7.5
      )
      assert msg =~ "Models not initialized"
    end
  end

  describe "vae_decode/1" do
    test "validates latents is a tensor" do
      assert {:error, msg} = SdxlServer.vae_decode("not_a_tensor")
      assert msg == "latents must be an Nx.Tensor"
    end

    test "accepts valid tensor" do
      latents = Nx.broadcast(0.5, {1, 4, 8, 8})
      assert {:error, msg} = SdxlServer.vae_decode(latents)
      assert msg =~ "Models not initialized"
    end
  end

  describe "vae_encode/1" do
    test "validates image is a tensor" do
      assert {:error, msg} = SdxlServer.vae_encode("not_a_tensor")
      assert msg == "image must be an Nx.Tensor"
    end

    test "accepts valid tensor" do
      image = Nx.broadcast(0.5, {1, 3, 64, 64})
      assert {:error, msg} = SdxlServer.vae_encode(image)
      assert msg =~ "Models not initialized"
    end
  end

  describe "generate_latents/3" do
    test "validates height is a positive integer" do
      assert {:error, msg} = SdxlServer.generate_latents(0, 1024)
      assert msg =~ "height must be a positive integer"

      assert {:error, msg} = SdxlServer.generate_latents(-1, 1024)
      assert msg =~ "height must be a positive integer"

      assert {:error, msg} = SdxlServer.generate_latents("1024", 1024)
      assert msg =~ "height must be a positive integer"
    end

    test "validates width is a positive integer" do
      assert {:error, msg} = SdxlServer.generate_latents(1024, 0)
      assert msg =~ "width must be a positive integer"

      assert {:error, msg} = SdxlServer.generate_latents(1024, -1)
      assert msg =~ "width must be a positive integer"

      assert {:error, msg} = SdxlServer.generate_latents(1024, "1024")
      assert msg =~ "width must be a positive integer"
    end

    test "validates seed when provided" do
      assert {:error, msg} = SdxlServer.generate_latents(1024, 1024, seed: -1)
      assert msg =~ "seed must be a non-negative integer or nil"

      assert {:error, msg} = SdxlServer.generate_latents(1024, 1024, seed: "42")
      assert msg =~ "seed must be a non-negative integer or nil"
    end

    test "accepts valid dimensions without seed" do
      assert {:error, msg} = SdxlServer.generate_latents(1024, 1024)
      assert msg =~ "Models not initialized"
    end

    test "accepts valid dimensions with seed" do
      assert {:error, msg} = SdxlServer.generate_latents(1024, 1024, seed: 42)
      assert msg =~ "Models not initialized"
    end
  end

  describe "get_model_info/0" do
    test "returns not initialized status when models not loaded" do
      assert {:ok, info} = SdxlServer.get_model_info()
      assert info.status == :not_initialized
    end
  end

  describe "clear_memory/0" do
    test "returns no model loaded status when models not loaded" do
      assert {:ok, info} = SdxlServer.clear_memory()
      assert info.status == :no_model_loaded
    end
  end
end
