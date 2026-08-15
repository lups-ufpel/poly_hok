defmodule OpenclBackend.MixProject do
  use Mix.Project

  def project do
    [
      app: :opencl_backend,
      version: "0.1.1",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      compilers: Mix.compilers() ++ [:cmake_compiler],
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
      poly_hok_dep(),
      {:cmake_compiler, git: "https://github.com/lups-ufpel/poly_hok.git", sparse: "cmake_compiler"}
    ]
  end

  defp poly_hok_dep() do
    if File.exists?("../../poly_hok") do
      IO.puts("[INFO] Using local poly_hok dependency (development use only!)")

      {:poly_hok, path: "../../poly_hok"}
    else
      {:poly_hok, git: "https://github.com/lups-ufpel/poly_hok.git", sparse: "poly_hok"}
    end
  end
end
