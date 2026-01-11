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
