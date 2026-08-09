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
  defk map_comp(a1,a2,resp,size,f) do
    index = blockIdx.x * blockDim.x + threadIdx.x
    stride = blockDim.x * gridDim.x

    for i in range(index,size,stride) do
          resp[i] = f(a1,a2,i)
    end
  end
 def comp_func(arr1,arr2,size,f) do
    d_arr1 = PolyHok.new_gnx(arr1)
    d_arr2 = PolyHok.new_gnx(arr2)
    shape = PolyHok.get_shape(d_arr1)
    type = PolyHok.get_type(d_arr1)
    result_gpu = PolyHok.new_gnx(shape,type)

    threadsPerBlock = 128;
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    PolyHok.spawn(&PMap.map_comp/5,
              {numberOfBlocks,1,1},
              {threadsPerBlock,1,1},
              [d_arr1,d_arr2,result_gpu,size, f])
    PolyHok.get_gnx result_gpu
 end
end

a = Nx.tensor(Enum.to_list(1..1000),type: {:s, 32})
b = Nx.tensor(Enum.to_list(1..1000),type: {:s, 32})

size = Tuple.product(Nx.shape(a))

host_resp = PolyHok.gpu_for n <- a,  do: n * n

IO.inspect host_resp

host_resp = PolyHok.gpu_for i <- 0..size, a, b, do:  2 * a[i] + b[i]

IO.inspect host_resp
