require PolyHok

PolyHok.set_debug_logs(true)
# PolyHok.TypeInference.set_debug_logs(true)

PolyHok.defmodule SimpleTest do
  defk simple_kernel(array, size, f) do
    index = blockIdx.x * blockDim.x + threadIdx.x

    if index < size do
      array[index] = f(array[index])
    end
  end
end

array_size = 100
array_cpu_int = Nx.tensor(Enum.to_list(1..array_size), type: {:s, 32})

IO.inspect(array_cpu_int, label: "CPU Array [int]")

# Create a tensor on the GPU copying the data from the CPU tensor
array_gpu_int = array_cpu_int |> PolyHok.new_gnx()

# Spawn the kernel to run on the GPU
PolyHok.spawn(
  # Kernel function
  &SimpleTest.simple_kernel/2,
  # Number of blocks
  {1, 1, 1},
  # Threads per block
  {array_size, 1, 1},
  # Kernel parameters
  [
    array_gpu_int,
    array_size,
    PolyHok.phok(fn x -> x * 2 end)
  ]
)

# Get result back to CPU
result_int = PolyHok.get_gnx(array_gpu_int)

IO.inspect(result_int, label: "Result after kernel execution [int]")
