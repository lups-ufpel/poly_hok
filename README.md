# PolyHok

PolyHok (Polymorphic Higher-Order GPU Kernels) is a DSL made for GPU programming in Elixir. PolyHok has several functional programming features to provide high expressiveness and ease of use, such as higher-order polymorphic kernels (i.e., functions that run on the GPU and can receive other functions as parameters), dynamic typing, and automatic device memory management.

<div align="center">
  <a href="https://lups.inf.ufpel.edu.br/polyhok">
    <img src="imgs/polyhok-logo.png" alt="PolyHok Logo" width="300"/>
  </a>
</div>

The PolyHok project is maintained by the [LUPS Research Group](https://lups.inf.ufpel.edu.br) at Universidade Federal de Pelotas (UFPel), Brazil. To know more about PolyHok, its documentation and features, please visit the [PolyHok website](https://lups.inf.ufpel.edu.br/polyhok).

## Usage

To use PolyHok in an Elixir project, you need to do three things:

1. Add PolyHok and Nx (version 0.9 or superior) as a dependency in your `mix.exs` file:

    ```elixir
    defp deps do
      [
        {:nx, "~> 0.9"},
        {:poly_hok, git: "https://github.com/lups-ufpel/poly_hok.git", sparse: "poly_hok"}
      ]
    end
    ```

2. Choose a PolyHok backend to use. Currently, PolyHok supports two backends: CUDA and OpenCL. You need to add one of them as a dependency in your `mix.exs` file. For example, to use the OpenCL backend, you can add the following line:

    ```elixir
    defp deps do
      [
        {:nx, "~> 0.9"},
        {:poly_hok, git: "https://github.com/lups-ufpel/poly_hok.git", sparse: "poly_hok"},
        {:opencl_backend, git: "https://github.com/lups-ufpel/poly_hok.git", sparse: "backends/opencl_backend"}
      ]
    end
    ```

    If you wish to use the CUDA backend, you can add the following line instead:

    ```elixir
    {:cuda_backend, git: "https://github.com/lups-ufpel/poly_hok.git", sparse: "backends/cuda_backend"}
    ```

3. Configure the backend in your application. PolyHok needs to know which backend to use at runtime. You can configure the backend in your `config/runtime.exs` file. For example, to use PolyHok with the OpenCL backend, your `config/runtime.exs` file should look like this:

    ```elixir
    import Config

    config :poly_hok, backend: OpenclBackend
    ```

## License

PolyHok is licensed under the [MIT License](LICENSE).
