# CmakeCompiler

A small `Mix.Task.Compiler` that invokes CMake as part of `mix compile`, with mtime-based staleness checking so it doesn't rebuild unless something has changed.

Built for [PolyHok](https://github.com/lups-ufpel/polyhok)'s GPU backends (`cuda_backend`, `opencl_backend`) so both can share the same CMake compiler logic instead of duplicating it.

## Installation

Add it as a dependency in your `mix.exs`:

```elixir
def deps do
  [
    {:cmake_compiler, git: "https://github.com/lups-ufpel/polyhok", sparse: "cmake_compiler"}
  ]
end
```

Then add `:cmake_compiler` to your project's compilers and configure the build, source directories and targets in your `mix.exs`:

```elixir
def project do
  [
    compilers: Mix.compilers() ++ [:cmake_compiler],
    cmake_build_dir: "CMakeBuild",
    cmake_source_dirs: ["c_src", "CMakeLists.txt"],
    cmake_targets: ["my_target.so"]
  ]
end
```

## Usage

Once added, `mix compile` will run CMake against your project's `CMakeLists.txt` **after** compiling your Elixir code, skipping the CMake step entirely if nothing has changed since the last build.

When you run `mix clean`, the CMake build directory will be removed, so the next `mix compile` will rebuild everything from scratch.

The configuration options are:

- `cmake_build_dir` (_optional_): The directory where CMake will generate its build files and compile the targets. Defaults to `CMakeBuild`. Must be a relative path to the project root and must be a **single** directory.

- `cmake_source_dirs` (_optional_): A list of directories and/or files that will be checked for staleness. If any of these files are newer than the last build, CMake will be invoked. Defaults to `["c_src", "CMakeLists.txt"]`. Must be relative paths to the project root.

- `cmake_targets` (**mandatory**): A list of targets that CMake will build. These targets will also be checked for staleness, so if any of them are missing or older than the source files, CMake will be invoked. Must be relative paths to the project root.

## Requirements

- CMake installed and on `PATH`
- A `CMakeLists.txt` at the root of the consuming project
- At least one target specified in the `cmake_targets` configuration
