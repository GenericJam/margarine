defmodule Margarine.MixProject do
  use Mix.Project

  def project do
    [
      app: :margarine,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      test_coverage: [tool: ExCoveralls],
      preferred_cli_env: [
        coveralls: :test,
        "coveralls.detail": :test,
        "coveralls.post": :test,
        "coveralls.html": :test
      ]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # Core dependencies (required)
      {:nx, "~> 0.9"},
      {:pythonx, "~> 0.2"},
      {:vix, "~> 0.31"},
      {:telemetry, "~> 1.0"},
      {:jason, "~> 1.4"},

      # Development and testing
      {:ex_doc, "~> 0.31", only: :dev, runtime: false},
      {:excoveralls, "~> 0.18", only: :test},
      {:dialyxir, "~> 1.4", only: [:dev, :test], runtime: false},

      # Optional backend dependencies (users choose one)
      # Uncomment the backend you want to use:
      # {:emlx, "~> 0.1", optional: true},
      # {:exla, "~> 0.9", optional: true},
      # {:torchx, "~> 0.7", optional: true}
    ]
  end
end
