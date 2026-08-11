require PolyHok

PolyHok.defmodule SimpleTest do
  defk simple_kernel(array, size) do
    index = blockIdx.x * blockDim.x + threadIdx.x

    if (index < size) do
      array[index] = array[index] + 1.0
    end
  end
end

array_size = 100

# Create a tensor on the CPU of type float with values from 1 to array_size
array_cpu = Nx.tensor(Enum.to_list(1..array_size), type: {:f, 32})

IO.inspect(array_cpu, label: "CPU Array")

# Create a tensor on the GPU copying the data from the CPU tensor
array_gpu = array_cpu |> PolyHok.new_gnx()

# Spawn the kernel to run on the GPU
PolyHok.spawn(
          &SimpleTest.simple_kernel/2,  # Kernel function
          {1, 1, 1},                    # Number of blocks
          {array_size, 1, 1},           # Threads per block
          [array_gpu, array_size])      # Kernel parameters

# Get result back to CPU
result = PolyHok.get_gnx(array_gpu)

IO.inspect(result, label: "Result after kernel execution")
