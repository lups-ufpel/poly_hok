defmodule Mix.Tasks.Compile.CmakeCompiler do
  use Mix.Task.Compiler

  # Here we are defining sensible defaults for the build and source directories,
  # but the user can override them in their Mix project configuration.
  @default_build_dir "CMakeBuild"
  @default_source_dirs ["c_src", "CMakeLists.txt"]

  @impl Mix.Task.Compiler
  def run(_args) do
    # Read the Mix project configuration, the user will put the desired values for the
    # CMake build there
    config = Mix.Project.config()

    build_dir = Keyword.get(config, :cmake_build_dir, @default_build_dir)
    source_dirs = Keyword.get(config, :cmake_source_dirs, @default_source_dirs)

    # At least one target must be specified in the Mix project configuration, otherwise we cannot
    # determine if the build is stale or not.
    targets = Keyword.fetch!(config, :cmake_targets)

    if stale?(source_dirs, targets) do
      Mix.shell().info("[#{app_name(config)} CMake Compiler] Configuring CMake build...")

      compiled_app_root = Mix.Project.app_path()

      Mix.shell().info("[#{app_name(config)} CMake Compiler] Compiled app root: #{compiled_app_root}")

      with :ok <- cmake(["-S", ".", "-B", build_dir, "-DCOMPILED_APP_ROOT=#{compiled_app_root}"]),
           :ok <- cmake(["--build", build_dir]) do
        :ok
      else
        {:error, msg} -> Mix.raise(msg)
      end
    else
      :noop
    end
  end

  @impl Mix.Task.Compiler
  def clean do
    config = Mix.Project.config()
    build_dir = Keyword.get(config, :cmake_build_dir, @default_build_dir)

    Mix.shell().info("[#{app_name(config)} CMake Compiler] Removing build directories...")
    File.rm_rf(build_dir)
    :ok
  end

  defp app_name(config), do: Keyword.fetch!(config, :app)

  defp stale?(source_dirs, targets) do
    target_mtimes = Enum.map(targets, &mtime/1)

    if Enum.any?(target_mtimes, &is_nil/1) do
      true
    else
      oldest_target = Enum.min(target_mtimes)

      newest_source =
        source_dirs |> Enum.flat_map(&source_files/1) |> Enum.map(&mtime/1) |> Enum.max()

      newest_source > oldest_target
    end
  end

  defp source_files(path) do
    cond do
      File.dir?(path) -> Path.wildcard(Path.join(path, "**/*"))
      File.regular?(path) -> [path]
      true -> []
    end
  end

  defp mtime(path) do
    case File.stat(path, time: :posix) do
      {:ok, %File.Stat{mtime: mtime}} -> mtime
      {:error, _} -> nil
    end
  end

  defp cmake(args) do
    case System.find_executable("cmake") do
      nil ->
        {:error, "cmake executable not found in PATH"}

      cmake_path ->
        case System.cmd(cmake_path, args, stderr_to_stdout: true, into: IO.stream()) do
          {_, 0} -> :ok
          {_, code} -> {:error, "cmake exited with status #{code}"}
        end
    end
  end
end
