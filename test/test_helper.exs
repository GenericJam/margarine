# Exclude integration tests by default
# They require model downloads and significant compute resources
ExUnit.start(exclude: [:integration])
