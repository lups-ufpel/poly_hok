defmodule Bmp.MixProject do
  use Mix.Project

  def project do
    [
      app: :bmp,
      version: "0.1.1",
      elixir: "~> 1.17",
      start_permanent: Mix.env() == :prod,
      deps: deps(),

      compilers: Mix.compilers() ++ [:cmake_compiler],
      cmake_build_dir: "CMakeBuild",
      cmake_source_dirs: ["c_src", "CMakeLists.txt"],
      cmake_targets: ["priv/bmp_nifs.so"]
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.9"},
      {:cmake_compiler, git: "https://github.com/lups-ufpel/poly_hok.git", sparse: "cmake_compiler"}
    ]
  end
end
