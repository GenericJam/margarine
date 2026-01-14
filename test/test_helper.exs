# Exclude integration tests by default
# They require model downloads and significant compute resources
# IMPORTANT: Integration tests MUST run with max_cases: 1 to prevent OOM
# Each test loads a ~12GB model into memory

# Suppress known benign Python multiprocessing warning
# This warning appears when Python processes are terminated by Elixir/BEAM
# See: https://github.com/apple/ml-stable-diffusion/issues/8
System.put_env("PYTHONWARNINGS", "ignore::UserWarning")

ExUnit.start(exclude: [:integration], max_cases: 1)
