require PolyHok

# PolyHok.set_debug_logs(true)
# PolyHok.TypeInference.set_debug_logs(true)

PolyHok.defmodule SimpleTest do
  defd menor(x, y) do
    if x < y do
      x
    else
      y
    end
  end

  defk simple_kernel(array_1, array_2, res_array, size, f) do
    index = blockIdx.x * blockDim.x + threadIdx.x

    if index < size do
      res_array[index] = f(array_1[index], array_2[index])
    end
  end
end

array_size = 100
array_cpu_1 = Nx.tensor(Enum.to_list(1..array_size), type: {:s, 32})
array_cpu_2 = Nx.tensor(Enum.to_list((array_size + 1)..(array_size * 2)), type: {:s, 32})

IO.inspect(array_cpu_1, label: "CPU Array 1 [int]")
IO.inspect(array_cpu_2, label: "CPU Array 2 [int]")

# Copying arrays to GPU
array_gpu_1 = array_cpu_1 |> PolyHok.new_gnx()
array_gpu_2 = array_cpu_2 |> PolyHok.new_gnx()

# Create empty GPU array to hold the result
array_gpu_res = PolyHok.new_gnx(PolyHok.get_shape(array_cpu_1), PolyHok.get_type(array_cpu_1))

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
    array_gpu_1,
    array_gpu_2,
    array_gpu_res,
    array_size,
    &SimpleTest.menor/2
  ]
)

# Get result back to CPU
result_cpu = PolyHok.get_gnx(array_gpu_res)

IO.inspect(result_cpu, label: "Result after kernel execution [int]")
