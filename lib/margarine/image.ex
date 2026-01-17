defmodule Margarine.Image do
  @moduledoc """
  Image encoding and decoding utilities for Margarine.

  Handles conversion between Nx tensors and PNG image format using Vix.

  ## Image Format

  Images are represented as Nx tensors with:
  - Type: `{:u, 8}` (unsigned 8-bit integers, 0-255)
  - Shape: `{height, width, 3}` (HWC format, RGB channels)
  - Or: `{batch, height, width, 3}` (BHWC format for batched images)

  ## Examples

      # Create a simple image tensor
      tensor = Nx.tensor([
        [[255, 0, 0], [0, 255, 0]],  # Red and green pixels
        [[0, 0, 255], [255, 255, 255]]  # Blue and white pixels
      ]) |> Nx.as_type(:u8)

      # Save as PNG
      Margarine.Image.save(tensor, "output.png")

      # Load PNG back
      {:ok, loaded} = Margarine.Image.load("output.png")

      # Convert tensor to PNG binary
      {:ok, png_binary} = Margarine.Image.from_nx(tensor)

      # Convert PNG binary back to tensor
      {:ok, tensor} = Margarine.Image.to_nx(png_binary)
  """

  alias Vix.Vips.Image, as: VixImage
  alias Vix.Vips.Operation

  @type tensor :: Nx.Tensor.t()
  @type png_binary :: binary()

  @doc """
  Converts an Nx tensor to PNG binary.

  ## Parameters

    * `tensor` - Nx tensor with shape `{H, W, 3}` or `{B, H, W, 3}` and type `{:u, 8}`

  ## Examples

      tensor = Nx.tensor([[[255, 0, 0]]]) |> Nx.as_type(:u8)
      {:ok, png_binary} = Margarine.Image.from_nx(tensor)

  """
  @spec from_nx(tensor()) :: {:ok, png_binary()} | {:error, String.t()}
  def from_nx(tensor) do
    with :ok <- validate_tensor(tensor) do
      # Normalize to 3D if needed
      normalized = normalize_shape(tensor)

      # Convert to Vix Image
      {height, width, _channels} = Nx.shape(normalized)

      # Vix expects data in HWC format (height, width, channels)
      # which matches our tensor format
      binary = Nx.to_binary(normalized)

      # Create Vix image from binary
      case VixImage.new_from_binary(binary, width, height, 3, :VIPS_FORMAT_UCHAR) do
        {:ok, vix_image} ->
          # Encode as PNG
          case VixImage.write_to_buffer(vix_image, ".png") do
            {:ok, png_binary} -> {:ok, png_binary}
            {:error, reason} -> {:error, "Failed to encode PNG: #{inspect(reason)}"}
          end

        {:error, reason} ->
          {:error, "Failed to create Vix image: #{inspect(reason)}"}
      end
    end
  end

  @doc """
  Converts PNG binary to Nx tensor.

  ## Parameters

    * `png_binary` - Binary data of a PNG image

  ## Examples

      {:ok, png_binary} = File.read("image.png")
      {:ok, tensor} = Margarine.Image.to_nx(png_binary)

  """
  @spec to_nx(png_binary()) :: {:ok, tensor()} | {:error, String.t()}
  def to_nx(png_binary) when is_binary(png_binary) do
    case VixImage.new_from_buffer(png_binary) do
      {:ok, vix_image} ->
        # Get image dimensions
        width = VixImage.width(vix_image)
        height = VixImage.height(vix_image)
        bands = VixImage.bands(vix_image)

        # Convert to RGB if needed (e.g., grayscale or RGBA)
        vix_image =
          cond do
            bands == 1 ->
              # Grayscale -> RGB
              {:ok, rgb} = Operation.bandjoin([vix_image, vix_image, vix_image])
              rgb

            bands == 4 ->
              # RGBA -> RGB (drop alpha channel)
              {:ok, rgb} = Operation.extract_band(vix_image, 0, n: 3)
              rgb

            bands == 3 ->
              # Already RGB
              vix_image

            true ->
              raise "Unsupported number of bands: #{bands}"
          end

        # Get binary data
        {:ok, binary} = VixImage.write_to_binary(vix_image)

        # Create Nx tensor
        tensor =
          binary
          |> Nx.from_binary(:u8)
          |> Nx.reshape({height, width, 3})

        {:ok, tensor}

      {:error, reason} ->
        {:error, "Failed to decode PNG: #{inspect(reason)}"}
    end
  end

  def to_nx(_invalid) do
    {:error, "Input must be a binary"}
  end

  @doc """
  Saves an Nx tensor as a PNG file.

  Creates parent directories if they don't exist.

  ## Parameters

    * `tensor` - Nx tensor with shape `{H, W, 3}` or `{B, H, W, 3}` and type `{:u, 8}`
    * `path` - File path to save to

  ## Examples

      tensor = Nx.tensor([[[255, 0, 0]]]) |> Nx.as_type(:u8)
      :ok = Margarine.Image.save(tensor, "/tmp/red_pixel.png")

  """
  @spec save(tensor(), String.t()) :: :ok | {:error, String.t()}
  def save(tensor, path) when is_binary(path) do
    with :ok <- validate_tensor(tensor),
         :ok <- ensure_parent_dir(path),
         {:ok, png_binary} <- from_nx(tensor),
         :ok <- File.write(path, png_binary) do
      :ok
    else
      {:error, reason} when is_binary(reason) -> {:error, reason}
      {:error, reason} -> {:error, "Failed to save image: #{inspect(reason)}"}
    end
  end

  @doc """
  Loads a PNG file as an Nx tensor.

  ## Parameters

    * `path` - File path to load from

  ## Examples

      {:ok, tensor} = Margarine.Image.load("/tmp/image.png")

  """
  @spec load(String.t()) :: {:ok, tensor()} | {:error, String.t()}
  def load(path) when is_binary(path) do
    case File.read(path) do
      {:ok, png_binary} ->
        to_nx(png_binary)

      {:error, :enoent} ->
        {:error, "File not found: #{path}"}

      {:error, reason} ->
        {:error, "Failed to read file: #{inspect(reason)}"}
    end
  end

  @doc """
  Validates that a tensor is suitable for image operations.

  Checks:
  - Type is `{:u, 8}`
  - Shape is either `{H, W, 3}` or `{B, H, W, 3}`
  - Dimensions are positive integers

  ## Examples

      tensor = Nx.tensor([[[255, 0, 0]]]) |> Nx.as_type(:u8)
      :ok = Margarine.Image.validate_tensor(tensor)

  """
  @spec validate_tensor(tensor()) :: :ok | {:error, String.t()}
  def validate_tensor(tensor) do
    # Check type first using pattern matching
    case Nx.type(tensor) do
      {:u, 8} ->
        # Type is correct, check shape
        validate_shape(tensor)

      type ->
        {:error, "Tensor must have type {:u, 8}, got: #{inspect(type)}"}
    end
  end

  defp validate_shape(tensor) do
    shape = Nx.shape(tensor)

    case tuple_size(shape) do
      3 ->
        # {H, W, C} format
        {_h, _w, c} = shape

        if c != 3 do
          {:error, "Tensor must have 3 channels (RGB), got: #{c} channels"}
        else
          :ok
        end

      4 ->
        # {B, H, W, C} format
        {_b, _h, _w, c} = shape

        if c != 3 do
          {:error, "Tensor must have 3 channels (RGB), got: #{c} channels"}
        else
          :ok
        end

      2 ->
        # 2D tensor - missing channel dimension
        {:error,
         "Tensor must have 3 channels (RGB), got 2-dimensional shape (missing channels): #{inspect(shape)}"}

      other ->
        {:error,
         "Tensor must have shape {H, W, 3} or {B, H, W, 3}, got #{other}-dimensional shape: #{inspect(shape)}"}
    end
  end

  @doc """
  Normalizes tensor shape to 3D (HWC) by removing batch dimension if present.

  ## Examples

      # 4D -> 3D
      tensor = Nx.tensor([[[[255, 0, 0]]]]) |> Nx.as_type(:u8)
      normalized = Margarine.Image.normalize_shape(tensor)  # {1, 1, 3}

      # 3D -> 3D (no change)
      tensor = Nx.tensor([[[255, 0, 0]]]) |> Nx.as_type(:u8)
      normalized = Margarine.Image.normalize_shape(tensor)  # {1, 1, 3}

  """
  @spec normalize_shape(tensor()) :: tensor()
  def normalize_shape(tensor) do
    case Nx.shape(tensor) do
      {_b, _h, _w, _c} ->
        # Remove batch dimension (assume batch size = 1)
        Nx.squeeze(tensor, axes: [0])

      {_h, _w, _c} ->
        # Already 3D
        tensor
    end
  end

  @doc """
  Preprocesses an image for VAE encoding.

  Takes an image tensor (HWC uint8) and prepares it for the VAE encoder:
  1. Resizes to target dimensions
  2. Converts from uint8 [0,255] to float32 [-1,1]
  3. Converts from HWC to BCHW format

  ## Parameters

    * `tensor` - Image tensor with shape `{H, W, 3}` and type `{:u, 8}`
    * `target_size` - Target `{height, width}` tuple

  ## Returns

    * `{:ok, preprocessed}` - Preprocessed tensor [1, 3, height, width] float32 in range [-1, 1]
    * `{:error, reason}` - Preprocessing failed

  ## Examples

      {:ok, image} = Margarine.Image.load("photo.png")
      {:ok, preprocessed} = Margarine.Image.preprocess_for_vae(image, {512, 512})
      # preprocessed is now [1, 3, 512, 512] float32 in range [-1, 1]

  """
  @spec preprocess_for_vae(tensor(), {pos_integer(), pos_integer()}) ::
          {:ok, tensor()} | {:error, String.t()}
  def preprocess_for_vae(tensor, {target_height, target_width})
      when is_integer(target_height) and is_integer(target_width) and
             target_height > 0 and target_width > 0 do
    with :ok <- validate_tensor(tensor) do
      # Normalize to 3D if needed
      tensor_3d = normalize_shape(tensor)
      {height, width, _channels} = Nx.shape(tensor_3d)

      # Resize if needed
      resized =
        if height != target_height or width != target_width do
          case resize(tensor_3d, target_height, target_width) do
            {:ok, resized_tensor} -> resized_tensor
            {:error, reason} -> throw({:error, reason})
          end
        else
          tensor_3d
        end

      # Convert uint8 [0, 255] -> float32 [-1, 1]
      # Formula: (pixel / 255.0) * 2.0 - 1.0
      normalized =
        resized
        |> Nx.as_type(:f32)
        |> Nx.divide(255.0)
        |> Nx.multiply(2.0)
        |> Nx.subtract(1.0)

      # Convert from HWC to BCHW: {H, W, 3} -> {1, 3, H, W}
      # Add batch dimension: {H, W, 3} -> {1, H, W, 3}
      batched = Nx.new_axis(normalized, 0)
      # Transpose to BCHW: {1, H, W, 3} -> {1, 3, H, W}
      bchw = Nx.transpose(batched, axes: [0, 3, 1, 2])

      {:ok, bchw}
    end
  catch
    {:error, reason} -> {:error, reason}
  end

  @doc """
  Resizes an image tensor to target dimensions using Vix.

  ## Parameters

    * `tensor` - Image tensor with shape `{H, W, 3}` and type `{:u, 8}`
    * `target_height` - Target height in pixels
    * `target_width` - Target width in pixels

  ## Returns

    * `{:ok, resized}` - Resized tensor with shape `{target_height, target_width, 3}`
    * `{:error, reason}` - Resize failed

  """
  @spec resize(tensor(), pos_integer(), pos_integer()) ::
          {:ok, tensor()} | {:error, String.t()}
  def resize(tensor, target_height, target_width)
      when is_integer(target_height) and is_integer(target_width) do
    {height, width, channels} = Nx.shape(tensor)

    # Ensure we have exactly 3 channels (RGB)
    if channels != 3 do
      {:error, "Tensor must have 3 channels (RGB), got #{channels}"}
    else
      # Strategy: Resize to approximately the right size using Vix (fast, high quality)
      # Then use Nx operations to crop/pad to exact dimensions (precise)

      # Calculate which dimension to match to minimize cropping/padding
      width_ratio = target_width / width
      height_ratio = target_height / height

      # Use the larger ratio to ensure we're at least as large as target in both dimensions
      scale = max(width_ratio, height_ratio)

      binary = Nx.to_binary(tensor)

      with {:ok, vix_image} <-
             VixImage.new_from_binary(binary, width, height, 3, :VIPS_FORMAT_UCHAR),
           # Resize using the scale factor
           {:ok, resized_vix} <- Operation.resize(vix_image, scale),
           {:ok, resized_binary} <- VixImage.write_to_binary(resized_vix) do
        # Get actual dimensions after resize
        actual_width = VixImage.width(resized_vix)
        actual_height = VixImage.height(resized_vix)
        actual_bands = VixImage.bands(resized_vix)

        # Convert back to Nx tensor
        resized_tensor =
          resized_binary
          |> Nx.from_binary(:u8)
          |> Nx.reshape({actual_height, actual_width, actual_bands})

        # Now adjust to exact target dimensions using Nx operations
        adjusted =
          resized_tensor
          |> maybe_crop_or_pad_height(actual_height, target_height)
          |> maybe_crop_or_pad_width(actual_width, target_width)

        {:ok, adjusted}
      else
        {:error, reason} -> {:error, "Resize failed: #{inspect(reason)}"}
      end
    end
  end

  # Crop or pad height to target
  defp maybe_crop_or_pad_height(tensor, current_height, target_height) do
    cond do
      current_height == target_height ->
        tensor

      current_height > target_height ->
        # Center crop
        start = div(current_height - target_height, 2)
        {_h, w, c} = Nx.shape(tensor)
        Nx.slice(tensor, [start, 0, 0], [target_height, w, c])

      current_height < target_height ->
        # Pad (bottom)
        {_h, _w, _c} = Nx.shape(tensor)
        pad_config = [{0, target_height - current_height}, {0, 0}, {0, 0}]
        Nx.pad(tensor, 0, pad_config)
    end
  end

  # Crop or pad width to target
  defp maybe_crop_or_pad_width(tensor, current_width, target_width) do
    cond do
      current_width == target_width ->
        tensor

      current_width > target_width ->
        # Center crop
        start = div(current_width - target_width, 2)
        {h, _w, c} = Nx.shape(tensor)
        Nx.slice(tensor, [0, start, 0], [h, target_width, c])

      current_width < target_width ->
        # Pad (right)
        {_h, _w, _c} = Nx.shape(tensor)
        pad_config = [{0, 0}, {0, target_width - current_width}, {0, 0}]
        Nx.pad(tensor, 0, pad_config)
    end
  end

  # Private helpers

  defp ensure_parent_dir(path) do
    path
    |> Path.dirname()
    |> File.mkdir_p()
    |> case do
      :ok -> :ok
      {:error, reason} -> {:error, "Failed to create directory: #{inspect(reason)}"}
    end
  end
end
