import Config

# Configure Pythonx to automatically install UV
# This ensures UV is available even on fresh machines
# Downloads happen at compile time and are bundled with releases
config :pythonx,
  uv_version: "0.8.5"

# Configure Nx backend - users should override this in their config
# For Apple Silicon: EMLX.Backend
# For NVIDIA/AMD GPU or CPU: EXLA.Backend
# config :nx, default_backend: EMLX.Backend
