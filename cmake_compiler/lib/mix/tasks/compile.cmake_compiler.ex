defmodule Mix.Tasks.Compile.CmakeCompiler do
  use Mix.Task.Compiler

  @build_dir "CMakeBuild"
  @source_dirs ["c_src", "CMakeLists.txt"]
  @targets ["priv/gpu_nifs.so", "priv/bmp_nifs.so"]

  @impl Mix.Task.Compiler
  def run(_args) do
    if stale?() do
      Mix.shell().info("[OpenCLBackend CMake Compiler] Configuring CMake build...")

      with :ok <- cmake(["-S", ".", "-B", @build_dir]),
           :ok <- cmake(["--build", @build_dir]) do
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
    Mix.shell().info("[OpenCLBackend CMake Compiler] Removing build and priv directories...")
    File.rm_rf(@build_dir)
    File.rm_rf("priv/")
    :ok
  end

  defp stale?() do
    target_mtimes = Enum.map(@targets, &mtime/1)

    if Enum.any?(target_mtimes, &is_nil/1) do
      # a target artifact is missing -> definitely need to build
      true
    else
      oldest_target = Enum.min(target_mtimes)

      newest_source =
        @source_dirs |> Enum.flat_map(&source_files/1) |> Enum.map(&mtime/1) |> Enum.max()

      newest_source > oldest_target
    end
  end

  defp source_files(path) do
    cond do
      # If it is a directory, recursively get all files in it
      File.dir?(path) -> Path.wildcard(Path.join(path, "**/*"))
      # If it is a regular file, return it as a single-element list
      File.regular?(path) -> [path]
      # Anything else return an empty list
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
