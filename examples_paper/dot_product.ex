require PolyHok
#Nx.default_backend(EXLA.Backend)
#import Nx
PolyHok.defmodule Ske do
include CAS
  defk map_2kernel(a1,a2,a3,size,f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    if(id < size) do
      a3[id] = f(a1[id],a2[id])
    end
  end
  def map2(t1,t2,func) do

    {l,c} = PolyHok.get_shape_gnx(t1)
    type = PolyHok.get_type_gnx(t2)
     size = l*c
     result_gpu = PolyHok.new_gnx(l,c, type)

      threadsPerBlock = 256;
      numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

      PolyHok.spawn(&Ske.map_2kernel/5,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[t1,t2,result_gpu,size,func])


      result_gpu
  end
  def reduce(ref, initial, f) do

     {l,c} = PolyHok.get_shape_gnx(ref)
     type = PolyHok.get_type_gnx(ref)
     size = l*c
      result_gpu  = PolyHok.new_gnx(Nx.tensor([[initial]] , type: type))

      threadsPerBlock = 256
      blocksPerGrid = div(size + threadsPerBlock - 1, threadsPerBlock)
      numberOfBlocks = blocksPerGrid
      PolyHok.spawn(&Ske.reduce_kernel/4,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref,result_gpu, f, size])
      result_gpu
  end
  defk reduce_kernel(a,  ref4, f,n) do

    __shared__ cache[256]

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    cacheIndex = threadIdx.x

    temp = ref4[0]

    while (tid < n) do
      temp = f(a[tid], temp)
      tid = blockDim.x * gridDim.x + tid
    end

    cache[cacheIndex] = temp
      __syncthreads()

    i = blockDim.x/2

    while (i != 0 ) do  ###&& tid < n) do
      #tid = blockDim.x * gridDim.x + tid
      if (cacheIndex < i) do
        cache[cacheIndex] = f(cache[cacheIndex + i] , cache[cacheIndex])
      end

    __syncthreads()
    i = i/2
    end

  if (cacheIndex == 0) do
    current_value = ref4[0]
    while(!(current_value == atomic_cas(ref4,current_value,f(cache[0],current_value)))) do
      current_value = ref4[0]
    end
  end
 end
  def dot_product(arr1,arr2) do
    arr1
    |> PolyHok.new_gnx
    |> Ske.map2(PolyHok.new_gnx(arr2), PolyHok.phok fn (a,b)->a * b end)
    |> Ske.reduce(0, PolyHok.phok fn (a,b)->a + b end)
    |> PolyHok.get_gnx
   end
   def replicate(n, x), do: (for _ <- 1..n, do: x)
end


n = 10000
arr1 = Nx.tensor([Ske.replicate(n,1)],type: {:f, 32})
arr2 = Nx.tensor([Enum.to_list(1..n)],type: {:f, 32})


host_resp = Ske.dot_product(arr1,arr2)

IO.inspect host_resp