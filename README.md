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

## Repository Structure

The PolyHok repository is structured as a _poncho_ project, which is a monorepo that contains multiple Elixir applications. The PolyHok "core" is located in the `poly_hok/` directory, while the backends are located in the `backends/` directory. Each backend is an independent Elixir application that is intended to be used alongside the PolyHok application.

Below we list all the applications in the PolyHok repository:

- `poly_hok/`: The core PolyHok application. Responsible for providing the PolyHok macros and functions for GNx creation, kernel spawning, type inferencem, and defines the BackendBehavior that all backends must implement.
- `backends/opencl_backend/`: The OpenCL backend for PolyHok.
- `backends/cuda_backend/`: The CUDA backend for PolyHok.
- `cmake_compiler/`: An Elixir compiler task used by the backends and Bmp module to compile their C++/CUDA code using CMake. The end user does not need to include this dependency to be able to use PolyHok, as it is already included in the backends and Bmp module.
- `tools/bmp/`: The Bmp module is a library that provides a simple interface for writing BMP images. It is used by some PolyHok examples in the [PolyHok Benchmarks](https://github.com/lups-ufpel/poly_hok_benchmarks). The end user can include this dependency in their project if they wish to use it, but it is not required for PolyHok to work.

## License

PolyHok is licensed under the [MIT License](LICENSE).
