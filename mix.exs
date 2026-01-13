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
      ],

      # Hex package metadata
      name: "Margarine",
      source_url: "https://github.com/yourorg/margarine",
      description: "AI-powered image generation for Elixir using FLUX and Stable Diffusion",
      package: package(),
      docs: docs()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger],
      mod: {Margarine.Application, []}
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
      {:credo, "~> 1.7", only: [:dev, :test], runtime: false}

      # Optional backend dependencies (users choose one)
      # Uncomment the backend you want to use:
      # {:emlx, "~> 0.1", optional: true},
      # {:exla, "~> 0.9", optional: true},
      # {:torchx, "~> 0.7", optional: true}
    ]
  end

  defp package do
    [
      name: "margarine",
      licenses: ["MIT"],
      links: %{
        "GitHub" => "https://github.com/yourorg/margarine",
        "Changelog" => "https://github.com/yourorg/margarine/blob/master/CHANGELOG.md"
      },
      files: ~w(
        lib
        priv
        test
        examples
        .formatter.exs
        mix.exs
        README.md
        LICENSE
        CHANGELOG.md
      ),
      maintainers: ["Margarine Contributors"]
    ]
  end

  defp docs do
    [
      main: "Margarine",
      extras: [
        "README.md",
        "CHANGELOG.md",
        "examples/README.md"
      ],
      groups_for_extras: [
        Examples: ~r/examples\/.*/
      ],
      groups_for_modules: [
        "Core API": [Margarine, Margarine.Pipeline],
        Configuration: [Margarine.Config],
        "Python Integration": [
          Margarine.Application,
          Margarine.Python.FluxServer
        ],
        Schedulers: [Margarine.Schedulers.FluxEuler],
        Utilities: [Margarine.Image, Margarine.Memory]
      ]
    ]
  end
end
