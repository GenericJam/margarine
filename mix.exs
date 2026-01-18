defmodule Margarine.MixProject do
  use Mix.Project

  def project do
    [
      app: :margarine,
      version: "0.2.2",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      test_coverage: [tool: ExCoveralls],

      # Hex package metadata
      name: "Margarine",
      source_url: "https://github.com/GenericJam/margarine",
      description: "AI-powered text-to-image and image-to-image generation for Elixir using FLUX and SDXL",
      package: package(),
      docs: docs()
    ]
  end

  # CLI configuration
  def cli do
    [
      preferred_envs: [
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
      extra_applications: [:logger],
      mod: {Margarine.Application, []}
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      # Core dependencies (required)
      {:nx, "~> 0.9"},
      # Has uv_init native_tls: true support
      # {:pythonx, git: "https://github.com/livebook-dev/pythonx.git", ref: "12ece4"},
      {:pythonx, "~> 0.4.7"},
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
      # {:exla, "~> 0.10", optional: true},
      # {:torchx, "~> 0.7", optional: true}
    ]
  end

  defp package do
    [
      name: "margarine",
      licenses: ["MIT"],
      links: %{
        "GitHub" => "https://github.com/GenericJam/margarine",
        "Changelog" => "https://github.com/GenericJam/margarine/blob/master/CHANGELOG.md"
      },
      files: ~w(
        lib
        priv
        test
        examples
        assets
        config
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
      logo: "assets/logo.png",
      extras: [
        "README.md",
        "CHANGELOG.md",
        "notebooks/flux_getting_started.livemd",
        "notebooks/sdxl_getting_started.livemd",
        "examples/README.md"
      ],
      groups_for_extras: [
        Guides: ~r/notebooks\/.*/,
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
