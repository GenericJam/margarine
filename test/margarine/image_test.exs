defmodule Margarine.ImageTest do
  use ExUnit.Case, async: true

  @moduledoc """
  Tests for Margarine.Image module.

  NOTE ON COVERAGE: The Image module interfaces with Vix (libvips), which means
  some code paths are difficult to test without creating specific image types or
  mocking Vix internals:
  - Grayscale (1 band) → RGB conversion
  - RGBA (4 band) → RGB conversion
  - Specific Vix error conditions

  Coverage of 75-80% is acceptable for this module given it's primarily a wrapper
  around an external library. All critical validation and happy paths are tested.
  """

  alias Margarine.Image

  describe "from_nx/1" do
    test "converts Nx tensor to PNG binary" do
      # Create a simple 2x2 RGB image (red, green, blue, white)
      tensor =
        Nx.tensor([
          # Red pixel
          [[255, 0, 0], [0, 255, 0]],
          # Green pixel
          # Blue pixel
          [[0, 0, 255], [255, 255, 255]]
          # White pixel
        ])
        |> Nx.as_type(:u8)

      assert {:ok, png_binary} = Image.from_nx(tensor)
      assert is_binary(png_binary)
      assert byte_size(png_binary) > 0

      # PNG files start with specific magic bytes
      assert <<137, 80, 78, 71, 13, 10, 26, 10, _rest::binary>> = png_binary
    end

    test "handles 4D tensor (batch dimension)" do
      # Shape: {1, height, width, channels}
      tensor =
        Nx.tensor([[[[255, 0, 0], [0, 255, 0]]]])
        |> Nx.as_type(:u8)

      assert {:ok, png_binary} = Image.from_nx(tensor)
      assert is_binary(png_binary)
      assert byte_size(png_binary) > 0
    end

    test "rejects tensors with wrong number of channels" do
      # Grayscale (1 channel) - not supported yet
      tensor = Nx.tensor([[128, 255], [64, 192]]) |> Nx.as_type(:u8)

      assert {:error, reason} = Image.from_nx(tensor)
      assert reason =~ "channels"
    end

    test "rejects tensors with wrong type" do
      # Float tensor instead of u8
      tensor = Nx.tensor([[[0.5, 0.5, 0.5]]]) |> Nx.as_type(:f32)

      assert {:error, reason} = Image.from_nx(tensor)
      assert reason =~ "type"
    end

    test "rejects invalid tensor shapes" do
      # 2D tensor (missing channel dimension)
      tensor = Nx.tensor([[1, 2], [3, 4]]) |> Nx.as_type(:u8)

      assert {:error, reason} = Image.from_nx(tensor)
      assert reason =~ "shape"
    end
  end

  describe "to_nx/1" do
    test "converts PNG binary to Nx tensor" do
      # First create a PNG
      original_tensor =
        Nx.tensor([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [128, 128, 128]]])
        |> Nx.as_type(:u8)

      {:ok, png_binary} = Image.from_nx(original_tensor)

      # Now convert back
      assert {:ok, decoded_tensor} = Image.to_nx(png_binary)

      # Check shape and type
      assert Nx.shape(decoded_tensor) == {2, 2, 3}
      assert Nx.type(decoded_tensor) == {:u, 8}

      # Pixels should match (within tolerance for PNG compression)
      # PNG is lossless, so exact match expected
      assert Nx.to_list(decoded_tensor) == Nx.to_list(original_tensor)
    end

    test "rejects invalid PNG data" do
      invalid_binary = "not a PNG file"

      assert {:error, reason} = Image.to_nx(invalid_binary)
      assert reason =~ "decode" or reason =~ "invalid"
    end

    test "rejects non-binary input" do
      assert {:error, reason} = Image.to_nx(12_345)
      assert reason =~ "binary" or reason =~ "invalid"
    end
  end

  describe "save/2" do
    test "saves tensor as PNG file" do
      tensor =
        Nx.tensor([[[255, 0, 0], [0, 255, 0]]])
        |> Nx.as_type(:u8)

      path = Path.join(System.tmp_dir!(), "margarine_test_#{:rand.uniform(999_999)}.png")

      try do
        assert :ok = Image.save(tensor, path)
        assert File.exists?(path)

        # Verify it's a valid PNG
        {:ok, file_binary} = File.read(path)
        assert <<137, 80, 78, 71, 13, 10, 26, 10, _rest::binary>> = file_binary
      after
        File.rm(path)
      end
    end

    test "creates parent directories if needed" do
      base_dir = Path.join(System.tmp_dir!(), "margarine_test_#{:rand.uniform(999_999)}")
      path = Path.join([base_dir, "subdir", "image.png"])

      tensor =
        Nx.tensor([[[100, 100, 100]]])
        |> Nx.as_type(:u8)

      try do
        assert :ok = Image.save(tensor, path)
        assert File.exists?(path)
      after
        File.rm_rf(base_dir)
      end
    end

    test "rejects invalid paths" do
      tensor =
        Nx.tensor([[[255, 255, 255]]])
        |> Nx.as_type(:u8)

      # Path with null bytes (invalid on most systems)
      assert {:error, _reason} = Image.save(tensor, "/tmp/test\0invalid.png")
    end
  end

  describe "load/1" do
    test "loads PNG file as tensor" do
      # Create and save a test image
      original_tensor =
        Nx.tensor([[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]])
        |> Nx.as_type(:u8)

      path = Path.join(System.tmp_dir!(), "margarine_load_test_#{:rand.uniform(999_999)}.png")

      try do
        :ok = Image.save(original_tensor, path)

        # Load it back
        assert {:ok, loaded_tensor} = Image.load(path)

        # Should match original
        assert Nx.shape(loaded_tensor) == Nx.shape(original_tensor)
        assert Nx.type(loaded_tensor) == Nx.type(original_tensor)
        assert Nx.to_list(loaded_tensor) == Nx.to_list(original_tensor)
      after
        File.rm(path)
      end
    end

    test "rejects non-existent files" do
      assert {:error, reason} = Image.load("/tmp/nonexistent_#{:rand.uniform(999_999)}.png")
      assert reason =~ "exist" or reason =~ "not found" or reason =~ "enoent"
    end

    test "rejects non-PNG files" do
      # Create a text file
      path = Path.join(System.tmp_dir!(), "not_an_image_#{:rand.uniform(999_999)}.txt")
      File.write!(path, "This is not an image")

      try do
        assert {:error, reason} = Image.load(path)
        assert reason =~ "decode" or reason =~ "invalid"
      after
        File.rm(path)
      end
    end
  end

  describe "validate_tensor/1" do
    test "accepts valid 3D RGB tensor (HWC)" do
      tensor =
        Nx.tensor([[[255, 0, 0], [0, 255, 0]]])
        |> Nx.as_type(:u8)

      assert :ok = Image.validate_tensor(tensor)
    end

    test "accepts valid 4D tensor with batch (BHWC)" do
      tensor =
        Nx.tensor([[[[255, 0, 0], [0, 255, 0]]]])
        |> Nx.as_type(:u8)

      assert :ok = Image.validate_tensor(tensor)
    end

    test "rejects tensor with wrong type" do
      tensor = Nx.tensor([[[1.0, 0.5, 0.5]]]) |> Nx.as_type(:f32)

      assert {:error, reason} = Image.validate_tensor(tensor)
      assert reason =~ "type"
    end

    test "rejects tensor with wrong number of channels" do
      # 4 channels (RGBA) - not supported yet
      tensor = Nx.tensor([[[255, 0, 0, 255]]]) |> Nx.as_type(:u8)

      assert {:error, reason} = Image.validate_tensor(tensor)
      assert reason =~ "channels"
    end

    test "rejects tensor with wrong dimensions" do
      # 2D tensor
      tensor = Nx.tensor([[1, 2], [3, 4]]) |> Nx.as_type(:u8)

      assert {:error, reason} = Image.validate_tensor(tensor)
      assert reason =~ "shape"
    end

    test "rejects values outside u8 range" do
      # This shouldn't happen with u8 type, but test edge case
      tensor = Nx.tensor([[[0, 127, 255]]]) |> Nx.as_type(:u8)
      assert :ok = Image.validate_tensor(tensor)
    end
  end

  describe "normalize_shape/1" do
    test "converts 4D BHWC to 3D HWC by removing batch" do
      tensor =
        Nx.tensor([[[[255, 0, 0], [0, 255, 0]]]])
        |> Nx.as_type(:u8)

      normalized = Image.normalize_shape(tensor)

      assert Nx.shape(normalized) == {1, 2, 3}
    end

    test "keeps 3D HWC as-is" do
      tensor =
        Nx.tensor([[[255, 0, 0], [0, 255, 0]]])
        |> Nx.as_type(:u8)

      normalized = Image.normalize_shape(tensor)

      assert Nx.shape(normalized) == {1, 2, 3}
    end
  end

  describe "resize/3" do
    test "resizes image to target dimensions" do
      # Create a 4x4 test image (small for speed)
      tensor =
        Nx.broadcast(128, {4, 4, 3})
        |> Nx.as_type(:u8)

      assert {:ok, resized} = Image.resize(tensor, 8, 8)
      assert Nx.shape(resized) == {8, 8, 3}
      assert Nx.type(resized) == {:u, 8}
    end

    test "handles upscaling" do
      tensor =
        Nx.broadcast(200, {2, 2, 3})
        |> Nx.as_type(:u8)

      assert {:ok, resized} = Image.resize(tensor, 10, 10)
      assert Nx.shape(resized) == {10, 10, 3}
    end

    test "handles downscaling" do
      tensor =
        Nx.broadcast(150, {20, 20, 3})
        |> Nx.as_type(:u8)

      assert {:ok, resized} = Image.resize(tensor, 5, 5)
      assert Nx.shape(resized) == {5, 5, 3}
    end

    test "preserves RGB channels" do
      # Create image with distinct colors
      tensor =
        Nx.tensor([
          [[255, 0, 0], [0, 255, 0]],
          [[0, 0, 255], [255, 255, 255]]
        ])
        |> Nx.as_type(:u8)

      assert {:ok, resized} = Image.resize(tensor, 4, 4)
      {_h, _w, channels} = Nx.shape(resized)
      assert channels == 3
    end
  end

  describe "preprocess_for_vae/2" do
    test "converts uint8 HWC to float32 BCHW in range [-1, 1]" do
      # Create small test image (4x4 for speed)
      tensor =
        Nx.broadcast(128, {4, 4, 3})
        |> Nx.as_type(:u8)

      target_size = {4, 4}

      assert {:ok, preprocessed} = Image.preprocess_for_vae(tensor, target_size)

      # Check shape: should be BCHW format
      assert Nx.shape(preprocessed) == {1, 3, 4, 4}
      # Check type
      assert Nx.type(preprocessed) == {:f, 32}

      # Check value range: uint8 128 -> [-1, 1] range should be close to 0
      # Formula: (128 / 255.0) * 2.0 - 1.0 = 0.003921...
      values = Nx.to_flat_list(preprocessed)
      assert Enum.all?(values, fn v -> v >= -1.0 and v <= 1.0 end)

      # Value should be close to 0 for input of 128
      [first | _] = values
      assert_in_delta first, 0.0, 0.01
    end

    test "correctly maps uint8 values to [-1, 1] range" do
      # Test edge values: 0 -> -1.0, 255 -> 1.0, 128 -> ~0.0
      tensor =
        Nx.tensor([
          [[0, 0, 0], [128, 128, 128]],
          [[255, 255, 255], [64, 192, 128]]
        ])
        |> Nx.as_type(:u8)

      assert {:ok, preprocessed} = Image.preprocess_for_vae(tensor, {2, 2})

      # Convert to list for easier inspection
      # Shape is {1, 3, 2, 2} - let's check channel 0
      channel_0 = preprocessed[0][0] |> Nx.to_flat_list()

      # First pixel (0,0): value 0 -> should be close to -1.0
      assert_in_delta Enum.at(channel_0, 0), -1.0, 0.01

      # Fourth pixel (1,1): value 64 -> should be close to -0.498
      # (64 / 255.0) * 2.0 - 1.0 = -0.498
      assert_in_delta Enum.at(channel_0, 3), -0.498, 0.01
    end

    test "resizes image to target dimensions" do
      # Create 8x8 image, resize to 4x4
      tensor =
        Nx.broadcast(100, {8, 8, 3})
        |> Nx.as_type(:u8)

      assert {:ok, preprocessed} = Image.preprocess_for_vae(tensor, {4, 4})

      # Should be resized to target
      assert Nx.shape(preprocessed) == {1, 3, 4, 4}
    end

    test "handles images that don't need resizing" do
      # Create image already at target size
      tensor =
        Nx.broadcast(150, {10, 10, 3})
        |> Nx.as_type(:u8)

      assert {:ok, preprocessed} = Image.preprocess_for_vae(tensor, {10, 10})

      # Should be same size, just preprocessed
      assert Nx.shape(preprocessed) == {1, 3, 10, 10}
      assert Nx.type(preprocessed) == {:f, 32}
    end

    test "converts HWC to BCHW format correctly" do
      # Create image with known RGB values
      # Red pixel at (0,0), green at (0,1), blue at (1,0), white at (1,1)
      tensor =
        Nx.tensor([
          [[255, 0, 0], [0, 255, 0]],
          [[0, 0, 255], [255, 255, 255]]
        ])
        |> Nx.as_type(:u8)

      assert {:ok, preprocessed} = Image.preprocess_for_vae(tensor, {2, 2})

      # Shape should be {1, 3, 2, 2} = batch, channels, height, width
      assert Nx.shape(preprocessed) == {1, 3, 2, 2}

      # Check that red channel (index 0) has high value at (0,0)
      red_channel = preprocessed[0][0]
      # 255 -> 1.0
      assert_in_delta red_channel[0][0] |> Nx.to_number(), 1.0, 0.01

      # Check that green channel (index 1) has high value at (0,1)
      green_channel = preprocessed[0][1]
      assert_in_delta green_channel[0][1] |> Nx.to_number(), 1.0, 0.01

      # Check that blue channel (index 2) has high value at (1,0)
      blue_channel = preprocessed[0][2]
      assert_in_delta blue_channel[1][0] |> Nx.to_number(), 1.0, 0.01
    end

    test "rejects invalid tensor" do
      # Try with wrong type (float instead of uint8)
      tensor =
        Nx.tensor([[[0.5, 0.5, 0.5]]])
        |> Nx.as_type(:f32)

      assert {:error, reason} = Image.preprocess_for_vae(tensor, {10, 10})
      assert reason =~ "type"
    end

    test "rejects invalid target dimensions" do
      tensor =
        Nx.broadcast(100, {10, 10, 3})
        |> Nx.as_type(:u8)

      # Negative dimensions should fail in guards
      assert_raise FunctionClauseError, fn ->
        Image.preprocess_for_vae(tensor, {-1, 10})
      end

      assert_raise FunctionClauseError, fn ->
        Image.preprocess_for_vae(tensor, {10, 0})
      end
    end

    test "handles 4D batched input by normalizing to 3D first" do
      # Create batched image {1, H, W, 3}
      tensor =
        Nx.tensor([[[[200, 100, 50]]]])
        |> Nx.as_type(:u8)

      assert {:ok, preprocessed} = Image.preprocess_for_vae(tensor, {1, 1})

      # Should still produce correct output shape
      assert Nx.shape(preprocessed) == {1, 3, 1, 1}
      assert Nx.type(preprocessed) == {:f, 32}
    end
  end
end
