require PolyHok

PolyHok.defmodule PMap do
  defk map_ker(a1,a2,size,f) do
    index = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(index,size,stride) do
          a2[i] = f(a1[i])
    end
  end
  defd inc(x) do
    x+1
  end
  def map(input, f) do
    shape = PolyHok.get_shape(input)
    type = PolyHok.get_type(input)
    result_gpu = PolyHok.new_gnx(shape,type)
    size = Tuple.product(shape)
    threadsPerBlock = 128;
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    PolyHok.spawn(&PMap.map_ker/4,
              {numberOfBlocks,1,1},
              {threadsPerBlock,1,1},
              [input,result_gpu,size, f])
    result_gpu
  end
end

n = 10000000
arr1 = Nx.tensor([Enum.to_list(1..n)],type: {:s, 32})
arr2 = Nx.tensor([Enum.to_list(1..n)],type: {:f, 32})
arr3 = Nx.tensor([Enum.to_list(1..n)],type: {:f, 64})

host_res1 = arr1
    |> PolyHok.new_gnx
    |> PMap.map(&PMap.inc/1)
    |> PolyHok.get_gnx

host_res2 = arr2
    |> PolyHok.new_gnx
    |> PMap.map(&PMap.inc/1)
    |> PolyHok.get_gnx

host_res3 = arr3
    |> PolyHok.new_gnx
    |> PMap.map(PolyHok.phok fn (x) -> x + 1 end)
    |> PolyHok.get_gnx

IO.inspect host_res1
IO.inspect host_res2
IO.inspect host_res3