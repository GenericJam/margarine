defmodule Margarine.Application do
  @moduledoc """
  OTP Application for Margarine.

  Handles:
  - Pythonx initialization with UV for automatic Python + dependency management
  - Supervision tree for model servers (added on-demand)
  - Environment checks and warm-up for production deployments
  """

  use Application
  require Logger

  @impl true
  def start(_type, _args) do
    # Initialize Pythonx with Python dependencies before starting any services
    Logger.info("[Margarine] Initializing Python environment via UV...")
    initialize_pythonx()

    # Start supervisor - children are added on-demand when models are loaded
    children = []

    opts = [strategy: :one_for_one, name: Margarine.Supervisor]
    Supervisor.start_link(children, opts)
  end

  @doc """
  Check the Python environment status.

  Useful for production warm-up to ensure Python environment is initialized
  before handling requests.

  ## Returns

  Map with environment information:
  - `pythonx_initialized`: boolean indicating if Pythonx is ready
  - `python_version`: Python version string or nil
  - `dependencies`: list of installed packages (future)

  ## Example

      # In your startup script or health check
      Margarine.Application.check_environment()
      #=> %{pythonx_initialized: true, python_version: "3.11.5"}
  """
  @spec check_environment() :: map()
  def check_environment do
    # Try a simple Pythonx eval to check if it's working
    code = """
    import sys
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    """

    case Pythonx.eval(code, %{}) do
      {:error, _reason} ->
        %{pythonx_initialized: false, python_version: nil}

      {_result, globals} ->
        version =
          case Map.get(globals, "python_version") do
            nil -> nil
            %Pythonx.Object{} = obj -> Pythonx.decode(obj)
            other -> other
          end

        %{pythonx_initialized: true, python_version: version}
    end
  rescue
    _ -> %{pythonx_initialized: false, python_version: nil}
  end

  # Private helpers

  defp initialize_pythonx do
    # Define Python dependencies via pyproject.toml
    # UV will automatically:
    # 1. Download and install Python (>=3.11)
    # 2. Create a virtual environment
    # 3. Install all dependencies
    # 4. Cache everything for subsequent runs
    pyproject_toml = """
    [project]
    name = "margarine"
    version = "0.2.2"
    requires-python = ">=3.11"
    dependencies = [
      "torch>=2.0.0",
      "diffusers>=0.21.0",
      "transformers>=4.30.0",
      "accelerate>=0.20.0",
      "safetensors>=0.3.1",
      "protobuf>=3.20.0",
      "sentencepiece>=0.1.99",
      "psutil>=5.9.0"
    ]
    """

    try do
      Pythonx.uv_init(pyproject_toml)
      Logger.info("[Margarine] ✓ Python environment initialized")
    rescue
      error ->
        Logger.error("[Margarine] Failed to initialize Python environment: #{inspect(error)}")
        Logger.error("""
        [Margarine] Pythonx should automatically install UV if needed.
        If you're seeing this error, please check:
        1. Internet connection (first run downloads UV + Python + dependencies)
        2. Disk space (~15GB required for models + dependencies)
        3. File permissions in ~/.cache/pythonx/
        """)
        reraise error, __STACKTRACE__
    end
  end
end
