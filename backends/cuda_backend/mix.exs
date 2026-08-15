defmodule CudaBackend.MixProject do
  use Mix.Project

  def project do
    [
      app: :cuda_backend,
      version: "0.1.1",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps()
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
      {:poly_hok, git: "https://github.com/lups-ufpel/poly_hok.git", sparse: "poly_hok"}
    ]
  end
end
