require PolyHok
require Integer
#Nx.default_backend(EXLA.Backend)
#import Nx
PolyHok.defmodule DP do
include CAS
  def reduce(ref, initial, f) do

     {l,c} = PolyHok.get_shape_gnx(ref)
     type = PolyHok.get_type_gnx(ref)
     size = l*c
      result_gpu  = PolyHok.new_gnx(Nx.tensor([[initial]] , type: type))

      threadsPerBlock = 256
      blocksPerGrid = div(size + threadsPerBlock - 1, threadsPerBlock)
      numberOfBlocks = blocksPerGrid
      PolyHok.spawn(&DP.reduce_kernel/5,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref,result_gpu, initial,f, size])
      result_gpu
  end
  defk reduce_kernel(a, ref4, initial,f,n) do

    __shared__ cache[256]

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    cacheIndex = threadIdx.x

    temp = initial

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
  def replicate(n, x), do: (for _ <- 1..n, do: x)
  def rep_change(n,x), do: rep_pos(n,x)
  def rep_pos(0,_x), do: []
  def rep_pos(n,x), do:  [x | rep_neg(n-1,x)]
  def rep_neg(0,_x), do: []
  def rep_neg(n,x), do:  [-x | rep_pos(n-1,x)]
  def new_dataset_nx_a(n), do: gen_nx_f(n,a_gen_new_dataset_nx_f(div(n,2),<<>>,<<>>))
  defp a_gen_new_dataset_nx_f(0,a1,a2), do: <<a1::binary,a2::binary>>
  defp a_gen_new_dataset_nx_f(size, a1,a2) do

    {ax,ay} = if (rem(size,2) == 0) do
                v = :rand.uniform(100)/1
                {v,-v}
              else
                v = :rand.uniform(100)/1
                {-v,v}
              end

    a_gen_new_dataset_nx_f(
        size - 1,
        <<a1::binary, ax::float-little-32>>,
        <<a2::binary, ay::float-little-32>>
    )
  end
  def new_dataset_nx_b(n), do: gen_nx_f(n,b_gen_new_dataset_nx_f(div(n,2),<<>>,<<>>))
  defp b_gen_new_dataset_nx_f(0,b1,b2), do: <<b1::binary,b2::binary>>
  defp b_gen_new_dataset_nx_f(size, b1,b2) do

    b = :rand.uniform(5)/1

    b_gen_new_dataset_nx_f(
        size - 1,
        <<b1::binary, b::float-little-32>>,
        <<b2::binary, b::float-little-32>>
    )
  end
  defp gen_nx_f(size,ref), do:  %Nx.Tensor{data: %Nx.BinaryBackend{ state: ref}, type: {:f,32}, shape: {1,size}, names: [nil,nil]}
end
#PolyHok.include [DP]

use Ske

[arg] = System.argv()

n = String.to_integer(arg)



vet1 = DP.new_dataset_nx_a(n)
vet2 = DP.new_dataset_nx_b(n)

prev = System.monotonic_time()

ref1 = PolyHok.new_gnx(vet1)

ref2 = PolyHok.new_gnx(vet2)


_result = ref1
    |> Ske.map(ref2, PolyHok.phok fn (a,b) -> a * b end)
    |> Ske.reduce(0.0,PolyHok.phok fn (a,b) -> a + b end)
    |> PolyHok.get_gnx

#IO.inspect result

next = System.monotonic_time()


IO.puts "PolyHok\t#{n}\t#{System.convert_time_unit(next-prev,:native,:millisecond)}"
